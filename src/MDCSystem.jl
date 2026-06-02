# ====================================================================
# --- Core MDC Structs ---
# ====================================================================

struct MDCSystem{T, C, V<:AbstractVector{T}}
    cost::C          # TransformedCost structure
    θ₀::V            # Initial parameter vector
    dθ₀::V            # Initial parameter direction
    momentum::T      # H parameter
end

struct MDCWorkspace{V}
    diff_θ::V
    grad_cache::V
    # We can add internal structural caches here later if needed to make cost completely allocation-free
end

function MDCWorkspace(sys::MDCSystem)
    return MDCWorkspace(
        similar(sys.dθ₀),
        similar(sys.θ₀)
    )
end

struct MDCSpan
    negative::Float64
    positive::Float64
end

struct MDCCurve{S}
    negative_sol::S
    positive_sol::S
end

# ====================================================================
# --- Initial Conditions Generator ---
# ====================================================================

# ====================================================================
# --- Corrected Initial Conditions Generator ---
# ====================================================================

function initialise_lambda(sys::MDCSystem, ws::MDCWorkspace)
    θ₀ = sys.θ₀
    dθ₀ = sys.dθ₀  # Your user-supplied initial parameter direction
    H = sys.momentum
    
    # Evaluate the cost function at the initial position.
    # We still use the workspace cache to hold the initial gradient calculation
    # just in case downstream steps or callbacks need it warm.
    C = sys.cost(θ₀, ws.grad_cache)
    
    # Safety Check: Total energy (H) must exceed potential energy (C) 
    # for kinetic energy (H - C) to be strictly positive.
    if H <= C
        error("Momentum parameter H ($H) must be strictly greater than initial cost C ($C).")
    end
   
   
    # Let's allocate a type-stable vector and compute it in-place
    λ₀ = similar(θ₀)
    @. λ₀ = (H - C) * dθ₀
    
    return λ₀
end

# ====================================================================
# --- ODE Vector Field Factory ---
# ====================================================================

function vectorfield(sys::MDCSystem, ws::MDCWorkspace)
    cost = sys.cost
    grad_cache = ws.grad_cache
    diff_θ = ws.diff_θ

    θ₀ = sys.θ₀
    H = sys.momentum
    N = length(θ₀)

    return function f!(du, u, p, t)
        θ = @view u[1:N]
        λ = @view u[N+1:end]

        dθ = @view du[1:N]
        dλ = @view du[N+1:end]

        @. diff_θ = θ - θ₀
        dist = sum(abs2, diff_θ)

        C = cost(θ, grad_cache)

        μ2 = (C - H) / 2.0

        λ_dot_λ = dot(λ, λ)
        λ_dot_diff = dot(λ, diff_θ)

        μ1 = dist > 1e-5 ? (λ_dot_λ - 4.0 * μ2^2) / λ_dot_diff : 0.0

        inv_2μ2 = 1.0 / (2.0 * μ2)

        @. dθ = (-λ + μ1 * diff_θ) * inv_2μ2

        dθ_norm = norm(dθ)
        if dθ_norm > 1e-8
            @. dθ /= dθ_norm
        end

        damping = dot(λ, dθ) / (H - C + 1e-8)

        @. dλ = (μ1 * dθ - grad_cache) * damping

        return nothing
    end
end

# ====================================================================
# --- Core Mathematical Residual Checker ---
# ====================================================================

"""
    mdc_dHdu_residual(sys::MDCSystem, u, t)

Computes the raw L1 numerical drift from the theoretical identity dHdu = 0.
This is allocation-free and operates directly on the raw state vector `u`.
"""
function mdc_dHdu_residual(sys::MDCSystem, u, t)
    N = length(sys.θ₀)
    H = sys.momentum
    θ₀ = sys.θ₀

    θ = @view u[1:N]
    λ = @view u[N+1:end]
    
    # Value-only evaluation to compute current potential energy cost
    C = sys.cost(θ)
    μ2 = (C - H) / 2.0

    # Avoid allocation when computing dot products
    λ_dot_λ = dot(λ, λ)
    λ_dot_diff = zero(eltype(u))
    @inbounds for i in 1:N
        λ_dot_diff += λ[i] * (θ[i] - θ₀[i])
    end
    
    # Singularity gate near the integration origin
    μ1 = abs(t) > 1e-3 ? (λ_dot_λ - 4.0 * μ2^2) / λ_dot_diff : 0.0
    inv_2μ2 = 1.0 / (2.0 * μ2)

    # Calculate unnormalised dθ norm in a single allocation-free loop
    dθ_norm_sq = zero(eltype(u))
    @inbounds for i in 1:N
        dθ_i = (-λ[i] + μ1 * (θ[i] - θ₀[i])) * inv_2μ2
        dθ_norm_sq += dθ_i * dθ_i
    end
    dθ_norm = sqrt(dθ_norm_sq)

    # Accumulate the L1 discrepancy: sum(abs(λ + 2 * μ2 * dθ_normalised))
    residual = zero(eltype(u))
    if dθ_norm > 1e-8
        @inbounds for i in 1:N
            dθ_i_normed = ((-λ[i] + μ1 * (θ[i] - θ₀[i])) * inv_2μ2) / dθ_norm
            residual += abs(λ[i] + 2.0 * μ2 * dθ_i_normed)
        end
    end
    
    return residual
end

# ====================================================================
# --- Callback Factories ---
# ====================================================================

"""
    mdc_momentum_readjustment(sys::MDCSystem; tol=1e-2)

Creates a DiscreteCallback that watches the dHdu numerical drift. If it 
exceeds `tol`, the costate (λ) is orthogonally projected back onto the 
energy manifold where the Hamiltonian derivative identity matches perfectly.
"""
function mdc_momentum_readjustment(sys::MDCSystem; tol=1e-3)
    N = length(sys.θ₀)
    H = sys.momentum
    θ₀ = sys.θ₀

    # 1. Condition: Evaluate if drift exceeds threshold
    condition = (u, t, integrator) -> begin
        res = mdc_dHdu_residual(sys, u, t)
        return res > tol
    end

    # 2. Affect: Force-recalculate the costate vector along the proper path gradient
    affect! = (integrator) -> begin
        u = integrator.u
        θ = @view u[1:N]
        λ = @view u[N+1:end]

        C = sys.cost(θ)
        μ2 = (C - H) / 2.0

        λ_dot_λ = dot(λ, λ)
        λ_dot_diff = zero(eltype(u))
        @inbounds for i in 1:N
            λ_dot_diff += λ[i] * (θ[i] - θ₀[i])
        end
        
        μ1 = abs(integrator.t) > 1e-3 ? (λ_dot_λ - 4.0 * μ2^2) / λ_dot_diff : 0.0
        inv_2μ2 = 1.0 / (2.0 * μ2)

        # Build dθ projection layout in-place directly onto the integrator's costate views
        # We reuse the trailing space of the state vector safely.
        dθ_norm_sq = zero(eltype(u))
        @inbounds for i in 1:N
            dθ_i = (-λ[i] + μ1 * (θ[i] - θ₀[i])) * inv_2μ2
            dθ_norm_sq += dθ_i * dθ_i
        end
        dθ_norm = sqrt(dθ_norm_sq)

        if dθ_norm > 1e-8
            @inbounds for i in 1:N
                dθ_i_normed = ((-λ[i] + μ1 * (θ[i] - θ₀[i])) * inv_2μ2) / dθ_norm
                # Force update back onto manifold: λ = -2 * μ2 * dθ_normalised
                u[N+i] = -2.0 * μ2 * dθ_i_normed
            end
        end
        return nothing
    end

    return DiscreteCallback(condition, affect!)
end


# ====================================================================
# --- Core Catastrophe Guard (Highly Recommended Default) ---
# ====================================================================

"""
    mdc_safety_callback(sys::MDCSystem; tol=1e-4)

Returns a DiscreteCallback that terminates integration if the cost `C` 
approaches or exceeds the total momentum `H`.
"""
function mdc_safety_callback(sys::MDCSystem; tol=1e-4)
    N = length(sys.θ₀)
    
    condition = (u, t, integrator) -> begin
        θ = @view u[1:N]
        C = sys.cost(θ) 
        return C >= (sys.momentum - tol) 
    end
    
    affect! = integrator -> begin
        @warn "MDC integration terminated at t = $(integrator.t): Cost exceeded momentum limit."
        terminate!(integrator)
    end
    
    return DiscreteCallback(condition, affect!)
end

# ====================================================================
# --- Parameter Bounds Factory ---
# ====================================================================

"""
    mdc_bounds_callback(ids, lbs, ubs)

Returns a DiscreteCallback that terminates integration if specified parameter 
indices fall outside lower bounds `lbs` or upper bounds `ubs`.
"""
function mdc_bounds_callback(ids::Vector{<:Integer}, lbs::Vector{<:Number}, ubs::Vector{<:Number})
    condition = (u, t, integrator) -> begin
        @inbounds for (i, idx) in enumerate(ids)
            val = u[idx]
            if val < lbs[i] || val > ubs[i]
                return true
            end
        end
        return false
    end

    affect! = integrator -> begin
        @warn "MDC integration halted: A parameter crossed specified boundary constraints at t = $(integrator.t)."
        terminate!(integrator)
    end

    return DiscreteCallback(condition, affect!)
end

# ====================================================================
# --- Logging / Verbose Factory ---
# ====================================================================

"""
    mdc_verbose_callbacks(sys::MDCSystem, timepoints; is_negative=false)

Returns a Tuple of PresetTimeCallbacks for logging the path arc-length and 
Hamiltonian energy drift (residual).
"""
function mdc_verbose_callbacks(sys::MDCSystem, timepoints; is_negative=false)
    N = length(sys.θ₀)
    t_points = is_negative ? -abs.(collect(timepoints)) : abs.(collect(timepoints))
    
    distance_cb = PresetTimeCallback(t_points, integrator -> begin
        @info "MDC: Curve arc-length reached t = $(integrator.t)"
    end)

    residual_cb = PresetTimeCallback(t_points, integrator -> begin
        u = integrator.u
        θ = @view u[1:N]
        λ = @view u[N+1:end]
        
        C = sys.cost(θ)
        current_H = C + 0.5 * sum(abs2, λ)
        residual = current_H - sys.momentum
        
        @info "MDC Stability: Residual (ΔH) = $residual at curve length t = $(integrator.t)"
    end)

    return (distance_cb, residual_cb)
end




# ====================================================================
# --- High Level Solve Interface ---
# ====================================================================

function MDCsolve(sys::MDCSystem; span=MDCSpan(-10.0, 10.0), mode=:fast, callback=nothing)
    ws = MDCWorkspace(sys)
    vf! = vectorfield(sys, ws)
    
    λ₀ = initialise_lambda(sys, ws)
    
    T = eltype(sys.θ₀)
    u0 = Vector{T}(undef, 2 * length(sys.θ₀))
    u0[1:length(sys.θ₀)] .= sys.θ₀
    u0[length(sys.θ₀)+1:end] .= λ₀
    
    alg = Tsit5()

    # Default to just the vital catastrophe safety guardrail if nothing is provided. GET RID 
    cb_to_use = isnothing(callback) ? mdc_safety_callback(sys) : callback

    sol_neg = nothing
    sol_pos = nothing

    if span.negative < 0.0
        prob_neg = ODEProblem(vf!, u0, (0.0, span.negative), sys)
        sol_neg = solve(prob_neg, alg, callback=cb_to_use)
    end

    if span.positive > 0.0
        prob_pos = ODEProblem(vf!, u0, (0.0, span.positive),sys)
        sol_pos = solve(prob_pos, alg, callback=cb_to_use)
    end

    return MDCCurve(sol_neg, sol_pos)
end

"""
    (curve::MDCCurve)(t::Real)

Enables continuous interpolation across the split-span trajectory. 
Automatically routes positive arc-lengths to `positive_sol` and negative 
arc-lengths to `negative_sol`.
"""
function (curve::MDCCurve)(t::Real)
    if t >= 0.0
        if isnothing(curve.positive_sol)
            error("Attempted to evaluate curve at t = $t, but positive_sol is uninitialized.")
        end
        return curve.positive_sol(t)
    else
        if isnothing(curve.negative_sol)
            error("Attempted to evaluate curve at t = $t, but negative_sol is uninitialized.")
        end
        # 🌟 FIX: Pass 't' directly as a negative number, 
        # because negative_sol's time domain natively spans from 0.0 down to -5.0!
        return curve.negative_sol(t)
    end
end
