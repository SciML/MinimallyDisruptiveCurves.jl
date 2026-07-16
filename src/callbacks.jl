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

abstract type AbstractController end

mutable struct PIController{T<:Real} <:AbstractController
    Kp::T
    Ki::T
    integral::Union{Nothing, Vector{T}}
end

function PIController(Kp::T, Ki::T) where {T}
    return PIController{T}(Kp, Ki, nothing)
end

"""
    mdc_continuous_momentum_readjustment(sys::MDCProblem, controller::AbstractController)

Creates a DiscreteCallback that implements PI control of the costate.
"""
function mdc_continuous_momentum_readjustment(sys::MDCProblem, controller::AbstractController)
    N = length(sys.θ₀)
    H = sys.momentum
    θ₀ = sys.θ₀

    # 2. Affect: adjust the costate vector towards the proper path gradient
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

        # calculate error
        error = zeros(eltype(u), N)
        if dθ_norm > 1.0e-8
            @inbounds for i in 1:N
                dθ_i_normed = ((-λ[i] + μ1 * (θ[i] - θ₀[i])) * inv_2μ2) / dθ_norm
                # calculate target position on manifold: λ = -2 * μ2 * dθ_normalised
                i_target = -2.0 * μ2 * dθ_i_normed
                error[i] = u[N + i] - i_target
            end
        end

        # update integral
        if isnothing(controller.integral)
            # First step: initialize integral term to zero vector
            controller.integral = zeros(eltype(u), N)
        else
            controller.integral += error
        end
        # Adjust costate
        u[(N + 1):end] -= (controller.Kp .* error) + (controller.Ki .* controller.integral)

        return nothing
    end

    return DiscreteCallback((u, t, integrator)->true, affect!)
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
