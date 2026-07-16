# ====================================================================
# --- Core MDC Structs ---
# ====================================================================

"""
    generate_fwd_caches(chain, θ₀)

Allocate reusable buffers for in-place forward transforms through `chain`.
"""
function generate_fwd_caches(chain::TransformChain, θ₀)
    return _generate_fwd_caches(chain.ts, θ₀)
end

@inline _generate_fwd_caches(::Tuple{}, θ₀) = ()
@inline function _generate_fwd_caches(ts::Tuple, θ₀)
    t = first(ts)
    rest = _mdc_tail(ts)
    out = forward(t, θ₀)
    return (similar(out), _generate_fwd_caches(rest, out)...)
end


abstract type AbstractMDCProblem end
abstract type AbstractMDCSolution end

"""
    MDCProblem(cost, theta0, dtheta0, momentum, names)

Holds and specifies the information needed to evolve an MD curve. names is an optional vector of symbols holding parameter names.
"""
struct MDCProblem{T, C, V <: AbstractVector{T}} <: AbstractMDCProblem
    cost::C          # TransformedCost structure
    θ₀::V            # Initial parameter vector
    dθ₀::V            # Initial parameter direction
    momentum::T
    names::Vector{Symbol}
end

function Base.show(io::IO, ::MIME"text/plain", sys::MDCProblem)
    print(io, "MDCProblem with ")
    print(io, "Parameters: ", length(sys.θ₀), ", ")
    return print(io, "Momentum cap (H): ", sys.momentum)
end

"""
    TransformedCost(core_cost::CostFunction)
    
"""
function TransformedCost(core_cost::CostFunction)
    return TransformedCost(core_cost, TransformChain())
end

function MDCProblem(raw_cost::CostFunction, θ₀, dθ₀, H; kwargs...)
    t_cost = TransformedCost(raw_cost)
    return MDCProblem(t_cost, θ₀, dθ₀, H; kwargs...)
end


function MDCProblem(cost, θ₀, dθ₀, momentum; names = [Symbol("θ_$i") for i in 1:length(θ₀)])
    return MDCProblem(cost, θ₀, dθ₀, momentum, names)
end

function MDCProblem(transformed_cost::TransformedCost, θ₀, dθ₀, momentum; names = nothing)
    N = length(θ₀)
    names = isnothing(names) ? [Symbol("θ_$i") for i in 1:N] : names
    # operational_names = transform_names(transformed_cost.chain, initial_names)

    return MDCProblem(transformed_cost, θ₀, dθ₀, momentum, names)
end

struct MDCWorkspace{V}
    diff_θ::V
    grad_cache::V
end

function MDCWorkspace(sys::MDCProblem)
    return MDCWorkspace(
        similar(sys.dθ₀),
        similar(sys.θ₀)
    )
end

"""
    MDCSpan(lower_bound <= 0, upper_bound >= 0)
Specifies the length of the MDC curve. negative values mean evolving th curve in the negative direction of the initial direction. If `lower_bound < 0` and `upper_bound > 0` then the curve evolution happens in two separate solves. 
"""
struct MDCSpan{T <: AbstractFloat}
    negative::T
    positive::T
end

"""
Object holding information on evolved curve. (c::MDCurve).spec gives the MDCProblem from which it was generated 
"""
struct MDCSolution{P, N, C <: AbstractMDCProblem} <: AbstractMDCSolution
    positive_sol::P
    negative_sol::N
    spec::C
end


"""
    Initialises costates based on momentum and initial param direction. Internal use.
    
"""
function initialise_lambda(sys::MDCProblem, ws::MDCWorkspace)
    θ₀ = sys.θ₀
    dθ₀ = sys.dθ₀  # Your user-supplied initial parameter direction
    H = sys.momentum
    C = sys.cost(θ₀, ws.grad_cache)
    if H <= C
        error("Momentum parameter H ($H) must be strictly greater than initial cost C ($C).")
    end


    λ₀ = similar(θ₀)
    @. λ₀ = (H - C) * dθ₀

    return λ₀
end

"""
    vectorfield(sys::MDCProblem)
Function factory to generate the vector field for the MDC  
"""
function vectorfield(sys::MDCProblem)
    cost = sys.cost
    θ₀ = sys.θ₀
    H = sys.momentum
    N = length(θ₀)
    chain = cost.chain
    fwd_caches = generate_fwd_caches(chain, θ₀)
    N_physical = isempty(fwd_caches) ? N : length(fwd_caches[end])

    # Allocate space caches once per thread closure
    grad_cache = Vector{eltype(θ₀)}(undef, N)
    diff_θ = Vector{eltype(θ₀)}(undef, N)
    gz_cache = Vector{eltype(θ₀)}(undef, N_physical)

    let grad_cache = grad_cache, gz_cache = gz_cache, diff_θ = diff_θ, N = N, cost = cost, H = H, θ₀ = θ₀, fwd_caches = fwd_caches
        return function f!(du, u, p, t)
            θ = @view u[1:N]
            λ = @view u[(N + 1):end]
            dθ = @view du[1:N]
            dλ = @view du[(N + 1):end]

            @. diff_θ = θ - θ₀
            dist = sum(abs2, diff_θ)

            C = cost(θ, grad_cache, gz_cache, fwd_caches)

            # --- MDC Core equations---
            μ2 = (C - H) / 2.0
            μ2_smooth = sign(μ2) * sqrt(μ2^2 + 1.0e-20)

            λ_dot_λ = dot(λ, λ)
            λ_dot_diff = dot(λ, diff_θ)

            μ1 = dist > 1.0e-5 ? (λ_dot_λ - 4.0 * μ2^2) / (λ_dot_diff + 1.0e-10 * sign(λ_dot_diff)) : 0.0
            inv_2μ2 = 1.0 / (2.0 * μ2)

            @. dθ = (-λ + μ1 * diff_θ) * inv_2μ2

            dθ_norm = norm(dθ)
            if dθ_norm > 1.0e-8
                @. dθ /= dθ_norm
            end

            energy_gap = max(1.0e-6, H - C)
            damping = dot(λ, dθ) / energy_gap

            @. dλ = (μ1 * dθ - grad_cache) * damping

            return nothing
        end
    end
end

# ====================================================================
# --- Callbacks for numerical stability ---
# ====================================================================

"""
    mdc_dHdu_residual(sys::MDCProblem, u, t)

Computes the raw L1 numerical drift from dHdu = 0.
This operates directly on the raw state vector `u`.
"""
function mdc_dHdu_residual(sys::MDCProblem, u, t)
    N = length(sys.θ₀)
    H = sys.momentum
    θ₀ = sys.θ₀

    θ = @view u[1:N]
    λ = @view u[(N + 1):end]

    C = sys.cost(θ)
    μ2 = (C - H) / 2.0

    λ_dot_λ = dot(λ, λ)
    λ_dot_diff = zero(eltype(u))
    @inbounds for i in 1:N
        λ_dot_diff += λ[i] * (θ[i] - θ₀[i])
    end

    # Singularity gate near the integration origin
    μ1 = abs(t) > 1.0e-3 ? (λ_dot_λ - 4.0 * μ2^2) / λ_dot_diff : 0.0
    inv_2μ2 = 1.0 / (2.0 * μ2)

    # Calculate unnormalised dθ norm
    dθ_norm_sq = zero(eltype(u))
    @inbounds for i in 1:N
        dθ_i = (-λ[i] + μ1 * (θ[i] - θ₀[i])) * inv_2μ2
        dθ_norm_sq += dθ_i * dθ_i
    end
    dθ_norm = sqrt(dθ_norm_sq)

    # Accumulate L1 discrepancy: sum(abs(λ + 2 * μ2 * dθ_normalised))
    residual = zero(eltype(u))
    if dθ_norm > 1.0e-8
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
    mdc_momentum_readjustment(sys::MDCProblem; tol=1e-2)

Creates a DiscreteCallback that watches the dHdu numerical drift. If it 
exceeds `tol`, the costate (λ) is orthogonally projected back onto the 
manifold where the Hamiltonian derivative identity holds.
"""
function mdc_momentum_readjustment(sys::MDCProblem; tol = 1.0e-3)
    N = length(sys.θ₀)
    H = sys.momentum
    θ₀ = sys.θ₀

    # 1. Condition: does drift exceed threshold
    condition = (u, t, integrator) -> begin
        res = mdc_dHdu_residual(sys, u, t)
        return res > tol
    end

    # 2. Affect: Force the costate vector onto the proper path gradient
    affect! = (integrator) -> begin
        u = integrator.u
        θ = @view u[1:N]
        λ = @view u[(N + 1):end]

        C = sys.cost(θ)
        μ2 = (C - H) / 2.0

        λ_dot_λ = dot(λ, λ)
        λ_dot_diff = zero(eltype(u))
        @inbounds for i in 1:N
            λ_dot_diff += λ[i] * (θ[i] - θ₀[i])
        end

        μ1 = abs(integrator.t) > 1.0e-3 ? (λ_dot_λ - 4.0 * μ2^2) / λ_dot_diff : 0.0
        inv_2μ2 = 1.0 / (2.0 * μ2)

        # Build dθ projection layout in-place directly onto the integrator's costate views
        # Reuse trailing space of state vector safely.
        dθ_norm_sq = zero(eltype(u))
        @inbounds for i in 1:N
            dθ_i = (-λ[i] + μ1 * (θ[i] - θ₀[i])) * inv_2μ2
            dθ_norm_sq += dθ_i * dθ_i
        end
        dθ_norm = sqrt(dθ_norm_sq)

        if dθ_norm > 1.0e-8
            @inbounds for i in 1:N
                dθ_i_normed = ((-λ[i] + μ1 * (θ[i] - θ₀[i])) * inv_2μ2) / dθ_norm
                # Force update back onto manifold: λ = -2 * μ2 * dθ_normalised
                u[N + i] = -2.0 * μ2 * dθ_i_normed
            end
        end
        return nothing
    end

    return DiscreteCallback(condition, affect!)
end

"""
    mdc_safety_callback(sys::MDCProblem; tol=1e-4)

Returns a DiscreteCallback that terminates integration if the cost `C` 
approaches or exceeds the total momentum `H`.
"""
function mdc_safety_callback(sys::MDCProblem; tol = 1.0e-4)
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


"""
    mdc_verbose_callbacks(sys::MDCProblem, timepoints; is_negative=false)

Returns a Tuple of PresetTimeCallbacks for logging the path arc-length and 
Hamiltonian energy drift (residual).
"""
function mdc_verbose_callbacks(sys::MDCProblem, timepoints; is_negative = false)
    N = length(sys.θ₀)
    t_points = is_negative ? -abs.(collect(timepoints)) : abs.(collect(timepoints))

    distance_cb = PresetTimeCallback(
        t_points, integrator -> begin
            @info "MDC: Curve arc-length reached t = $(integrator.t)"
        end
    )

    residual_cb = PresetTimeCallback(
        t_points, integrator -> begin
            u = integrator.u
            θ = @view u[1:N]
            λ = @view u[(N + 1):end]

            C = sys.cost(θ)
            current_H = C + 0.5 * sum(abs2, λ)
            residual = current_H - sys.momentum

            @info "MDC Stability: Residual (ΔH) = $residual at curve length t = $(integrator.t)"
        end
    )

    return (distance_cb, residual_cb)
end

"""
    MDCSolve(sys::MDCProblem; span, mode, dt, callback, parallel, alg=Tsit5()) -> MDCSolution

Solve the Minimally Disruptive Curve (MDC) differential equations for a given system.
This function integrates the MDC vector field both forwards and backwards in time from the 
initial state, returning an `MDCSolution` containing the joint trajectory pieces. It internally 
constructs a unified initial condition state vector combining parameters \$\\theta_0\$ and 
their respective tracking sensitivities \$\\lambda_0\$.

# Keyword arguments
- `span::MDCSpan`: arc-length range to integrate over (default `MDCSpan(-10.0, 10.0)`).
- `alg`: any `OrdinaryDiffEq.AbstractODEAlgorithm`. Defaults to `Tsit5()`. Pass a stiff
  solver (e.g. `Rodas5()`) if your cost function has stiff Jacobians.
- `mode`: `:adaptive` (default), `:fixed`, or `:fast`.
- `dt`: step size hint for `:fixed` and `:fast` modes.
- `callback`: a `DiscreteCallback`, `CallbackSet`, or `nothing`.
- `parallel::Bool`: solve the forward and backward pieces concurrently via `Threads.@spawn`.
"""
function MDCSolve(
        sys::MDCProblem;
        span = MDCSpan(-10.0, 10.0),
        mode = :adaptive,
        dt = 0.01,
        callback = nothing,
        parallel = false,
        alg = Tsit5()
    )

    ws = MDCWorkspace(sys)
    λ₀ = initialise_lambda(sys, ws)

    # 2. Build the unified initial conditions vector [θ₀; λ₀]
    T = eltype(sys.θ₀)
    u0 = Vector{T}(undef, 2 * length(sys.θ₀))
    u0[1:length(sys.θ₀)] .= sys.θ₀
    u0[(length(sys.θ₀) + 1):end] .= λ₀


    solve_kwargs = if mode == :fixed
        (adaptive = false, dt = dt)
    elseif mode == :fast
        (
            adaptive = true,
            force_dtmin = true,
            dt = dt,
            dtmin = 1.0e-6,
            unstable_check = (dt, u, p, t) -> false,
        )
    else
        (adaptive = true,)
    end

    run_neg() = begin
        if span.negative < 0.0
            local_vf_neg! = vectorfield(sys)
            prob_neg = ODEProblem(local_vf_neg!, u0, (0.0, span.negative), sys)
            return solve(prob_neg, alg; callback = callback, solve_kwargs...)
        end
        return nothing
    end

    run_pos() = begin
        if span.positive > 0.0
            local_vf_pos! = vectorfield(sys)
            prob_pos = ODEProblem(local_vf_pos!, u0, (0.0, span.positive), sys)
            return solve(prob_pos, alg; callback = callback, solve_kwargs...)
        end
        return nothing
    end

    sol_neg, sol_pos = if parallel
        t_neg = Threads.@spawn run_neg()
        t_pos = Threads.@spawn run_pos()
        fetch(t_neg), fetch(t_pos)
    else
        run_neg(), run_pos()
    end

    return MDCSolution(sol_pos, sol_neg, sys)
end

"""
    (curve::MDCSolution)(t::Real)
Enables continuous interpolation across the split-span trajectory. 
Routes positive arc-lengths to `positive_sol` and negative 
arc-lengths to `negative_sol`.
"""
function (curve::MDCSolution)(t::Real; type = :all)
    # 1. Null-Safe State Extraction
    raw_state = if t >= 0.0
        if !isnothing(curve.positive_sol)
            curve.positive_sol(t)
        elseif !isnothing(curve.negative_sol)
            curve.negative_sol(0.0)
        else
            error("Cannot evaluate MDCSolution: both positive and negative solutions are empty.")
        end
    else # t < 0.0
        if !isnothing(curve.negative_sol)
            curve.negative_sol(t)
        elseif !isnothing(curve.positive_sol)
            # Boundary stitch point: if backward is missing,
            # fall back to the initial state at the start of the forward path
            curve.positive_sol(0.0)
        else
            error("Cannot evaluate MDCSolution: both positive and negative solutions are empty.")
        end
    end

    if type == :all
        return raw_state
    end

    N_params = length(raw_state) ÷ 2
    if type == :parameters || type == :states
        return raw_state[1:N_params]
    elseif type == :costates
        return raw_state[(N_params + 1):end]
    else
        error("Unknown type filter: :$type. Use :all, :parameters, or :costates.")
    end
end

"""
    cost_trajectory(curve[, ts])

Evaluate the curve's cost along the supplied time grid, or along saved solution times.
"""
function cost_trajectory(curve::MDCSolution, ts::AbstractVector)
    cost = curve.spec.cost
    return [cost(curve(t; type = :parameters)) for t in ts]
end

function cost_trajectory(curve::MDCSolution)
    ts = if !isnothing(curve.negative_sol) && !isnothing(curve.positive_sol)
        vcat(curve.negative_sol.t, curve.positive_sol.t)
    elseif !isnothing(curve.negative_sol)
        curve.negative_sol.t
    elseif !isnothing(curve.positive_sol)
        curve.positive_sol.t
    else
        error("Cannot compute cost_trajectory: both solutions are empty.")
    end
    return cost_trajectory(curve, ts)
end


function Base.show(io::IO, ::MIME"text/plain", curve::MDCSolution)
    if isnothing(curve.spec)
        print(io, "Empty MDCSolution (uninitialized).")
        return
    end

    sys = curve.spec
    N_params = length(sys.θ₀)

    neg_max = !isnothing(curve.negative_sol) ? abs(minimum(curve.negative_sol.t)) : 0.0
    pos_max = !isnothing(curve.positive_sol) ? maximum(curve.positive_sol.t) : 0.0

    println(io, "Minimally Disruptive Curve (MDCSolution)")
    println(io, "====================================")
    println(io, "  • Parameter Dimensions : ", N_params)
    println(io, "  • Explored Span        : [", -neg_max, " ↔ ", pos_max, "] (Arc length)")
    println(io, "  • Initial Cost (C₀)    : ", round(sys.cost(sys.θ₀), digits = 5))
    println(io, "  • Total Energy (H)     : ", sys.momentum)

    state_pos = !isnothing(curve.positive_sol) ? curve.positive_sol.u[end][1:N_params] : sys.θ₀
    state_neg = !isnothing(curve.negative_sol) ? curve.negative_sol.u[end][1:N_params] : sys.θ₀

    #transform the baseline names on-the-fly.
    display_names = transform_names(sys.cost.chain, sys.names)

    println(io, "  • Max Parameter Shifts :")
    for i in 1:N_params
        p_name = (isassigned(display_names, i) && !isnothing(display_names[i])) ? string(display_names[i]) : "θ_$i"
        println(io, "      ", rpad(p_name, 20), ": [", round(state_neg[i], digits = 3), " ↔ ", round(state_pos[i], digits = 3), "]")
    end
    return
end
