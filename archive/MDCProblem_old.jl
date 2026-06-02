"""
    MDCProblem(cost, p0, dp0, momentum, tspan)

Creates an MDCProblem, that can then generate a minimally disruptive curve using evolve(c::MDCProblem, ...; ...)
"""
struct MDCProblem{A, B, C, D, E} <: CurveProblem
    cost::A
    p0::B
    dp0::C
    momentum::D
    tspan::E
    ## reverse initial direction and signflip curve span if the latter is nonpositive
    function MDCProblem(
            a::A, b::B, c::C, d::D, e::E
        ) where {A} where {B} where {C} where {D} where {E}
        if max(e...) <= 0.0
            e = map(x -> -x |> abs, e) |> reverse
            c = -c
        end
        return new{A, B, C, D, E}(a, b, c, d, e)
    end
end
isjumped(c::MDCProblem) = ZeroStart()
whatdynamics(c::MDCProblem) = MDCDynamics()

num_params(c::CurveProblem) = length(c.p0)
param_template(c::CurveProblem) = deepcopy(c.p0)
initial_params(c::CurveProblem) = c.p0



"""
    Callback to readjust momentum in the case that the numerical residual from the identity dHdu = 0 crosses a user-specified threshold
"""
function (m::MomentumReadjustment)(c::CurveProblem)
    return readjustment(c, ResidualCondition(), CostateAffect(), m.tol, m.verbose)
end
"""
    Callback to readjust state in the case that the numerical residual from the identity dHdu = 0 crosses a user-specified threshold. EXPERIMENTAL AND WILL PROBABLY BREAK
"""
function (m::StateReadjustment)(c::CurveProblem)
    return readjustment(c, ResidualCondition(), StateAffect(), m.tol, m.verbose)
end

"""
    (c::MDCProblem)()

returns a tuple of ODEProblems specified by the MDCProblem. Usually a single ODEProblem. Two are provided if the curve crosses zero, so that one can run two curves in parallel going backwards/forwards from zero
"""
function (c::MDCProblem)()
    spans = make_spans(c, c.tspan)
    cs = map(spans) do span
        mult = sign(span[end])
        return MDCProblem(c.cost, c.p0, mult * c.dp0, c.momentum, abs.(span))
    end
    spans = map(x -> abs.(x), spans)
    u0s = initial_conditions.(cs)
    u0 = map(span -> initial_conditions(c), spans)
    fs = dynamics.(cs)

    return ODEProblem.(fs, u0s, spans)
    # return map(sp -> ODEProblem(f, u0, sp), spans)  # make two problems for 2-sided tspan
end

"""
    make_spans(c::MDCProblem, span)

  - makes sure span of curve is increasing.
  - if the span crosses zero, then returns two separate spans. evolve then runs two curves in parallel, going backwards/forwards from zero.
"""
function make_spans(c::CurveProblem, span, ::ZeroStart)
    (span[1] > span[2]) && error("make your curve span monotone increasing")
    if (span[2] > 0) && (span[1] < 0)
        spans = ((0.0, span[1]), (0.0, span[2]))
    else
        spans = (span,)
    end
    return spans
end

"""
    initial_costate(c::MDCProblem)

solves for the initial costate required to evolve a MD curve.
"""
function initial_costate(c::MDCProblem)
    μ₂ = (-c.momentum + c.cost(c.p0)) / 2.0
    λ₀ = -2.0 * μ₂ * c.dp0
    return λ₀
end

"""
Generate initial conditions of an MDCProblem
"""
function initial_conditions(c::MDCProblem)
    λ₀ = initial_costate(c)
    return cat(c.p0, λ₀, dims = 1)
end

"""
Generate vector field for MD curve, as specified by MDCProblem
"""
function dynamics(c::CurveProblem, ::MDCDynamics)
    cost = c.cost
    ∇C = param_template(c)
    N = num_params(c)
    H = c.momentum
    θ₀ = initial_params(c)
    # Pre-allocate temporary array for θ - θ₀ computation
    diff_θ = similar(θ₀)
    function upd(du, u, p, t)
        θ = @view u[1:N] # current parameter vector (view, no allocation)
        λ = @view u[(N + 1):end] # current costate vector (view, no allocation)
        dθ = @view du[1:N]
        dλ = @view du[(N + 1):end]

        # Compute θ - θ₀ in-place
        @. diff_θ = θ - θ₀
        dist = sum(abs2, diff_θ) # Use abs2 instead of .^ 2 for efficiency
        C = cost(θ, ∇C) # also updates ∇C as a mutable
        μ2 = (C - H) / 2

        # Compute dot products without allocation
        λ_dot_λ = dot(λ, λ)
        λ_dot_diff = dot(λ, diff_θ)
        μ1 = dist > 1.0e-3 ? (λ_dot_λ - 4 * μ2^2) / λ_dot_diff : 0.0

        # Compute dθ in-place
        inv_2μ2 = 1 / (2 * μ2)
        @. dθ = (-λ + μ1 * diff_θ) * inv_2μ2

        # Normalize dθ in-place
        dθ_norm = sqrt(sum(abs2, dθ))
        @. dθ /= dθ_norm

        # Compute damping constant
        damping_constant = dot(λ, dθ) / (H - C)

        # Compute dλ in-place
        @. dλ = (μ1 * dθ - ∇C) * damping_constant

        return nothing
    end
    return upd
end

"""
Callback to stop MD Curve evolving if cost > momentum
"""
function (t::TerminalCond)(c::CurveProblem)
    cost = c.cost
    H = c.momentum
    N = num_params(c)
    function condition(u, t, integrator)
        θ = @view u[1:N]
        return (cost(θ) > H)
    end
    return DiscreteCallback(condition, SciMLBase.terminate!)
end

function readjustment(
        c::CurveProblem, cnd::ConditionType, aff::AffectType, momentum_tol, verbose::Bool
    )
    if isnan(momentum_tol)
        return nothing
    end
    cond = build_cond(c, cnd, momentum_tol)
    affect! = build_affect(c, aff)
    return DiscreteCallback(cond, affect!)
end

function build_cond(c::CurveProblem, ::ResidualCondition, tol, ::MDCDynamics)
    function rescond(u, t, integ)
        absres = dHdu_residual(c, u, t, integ)
        return absres > tol ? begin
                # @info "applying readjustment at t=$t, |res| = $absres"
                return true
            end : return false
    end
    return rescond
end

function build_cond(c::CurveProblem, ::CostCondition, tol)
    N = num_params(c)
    function costcond(u, t, integ)
        θ = @view u[1:N]
        return c.cost(θ) > tol
    end
    return costcond
end

"""
    For dHdu_residual and build_affect(::MDCProblem, ::CostateAffect): there is an unnecessary allocation in the line `dθ = ...`. I initially used `dθ[:] = ....`, but this produced unreliable output (the MDCurve changed on each solution). I found that this was because temporary arrays like this are not safe in callbacks, for some reason. The solution is to use SciMLBase.get_tmp_cache. Don't have time to figure out how to do this right now. Do at some point. 
"""

"""
    dHdu_residual(c::MDCProblem, u, t, dθ)

Checks dHdu residual (u deriv of Hamiltonian). Returns true if abs(residual) is greater than some tolerance (it should be zero)
"""
function dHdu_residual(c::CurveProblem, u, t, integ, ::MDCDynamics)
    N = num_params(c)
    H = c.momentum
    θ₀ = initial_params(c)

    θ = @view u[1:N]
    λ = @view u[(N + 1):end]
    μ2 = (c.cost(θ) - H) / 2.0

    # Compute dot products without allocations
    λ_dot_λ = dot(λ, λ)
    diff_sum = zero(eltype(u))
    λ_dot_diff = zero(eltype(u))
    @inbounds for i in 1:N
        diff_i = θ[i] - θ₀[i]
        λ_dot_diff += λ[i] * diff_i
    end
    μ1 = t > 1.0e-3 ? (λ_dot_λ - 4 * μ2^2) / λ_dot_diff : 0.0

    # Compute residual without allocating dθ - we compute the sum directly
    inv_2μ2 = 1 / (2 * μ2)
    dθ_norm_sq = zero(eltype(u))
    @inbounds for i in 1:N
        diff_i = θ[i] - θ₀[i]
        dθ_i = (-λ[i] + μ1 * diff_i) * inv_2μ2
        dθ_norm_sq += dθ_i * dθ_i
    end
    dθ_norm = sqrt(dθ_norm_sq)

    # Compute residual sum: sum(abs(λ + 2 * μ2 * dθ_normalized))
    residual = zero(eltype(u))
    @inbounds for i in 1:N
        diff_i = θ[i] - θ₀[i]
        dθ_i = (-λ[i] + μ1 * diff_i) * inv_2μ2 / dθ_norm
        residual += abs(λ[i] + 2 * μ2 * dθ_i)
    end
    return residual
end

"""
    build_affect(c::MDCProblem, ::CostateAffect)

Resets costate to undo effect of cumulative numerical error. Specifically, finds costate so that dHdu = 0, where H is the Hamiltonian.
"""
function build_affect(c::CurveProblem, ::CostateAffect, ::MDCDynamics)
    N = num_params(c)
    H = c.momentum
    θ₀ = initial_params(c)
    # Pre-allocate temporary array for dθ computation
    dθ_temp = param_template(c)
    diff_θ = param_template(c)
    function reset_costate!(integ)
        θ = @view integ.u[1:N]
        λ = @view integ.u[(N + 1):end]
        λ_out = @view integ.u[(N + 1):end]

        μ2 = (c.cost(θ) - H) / 2

        # Compute diff_θ and dot products without allocations
        @. diff_θ = θ - θ₀
        λ_dot_λ = dot(λ, λ)
        λ_dot_diff = dot(λ, diff_θ)
        μ1 = integ.t > 1.0e-3 ? (λ_dot_λ - 4 * μ2^2) / λ_dot_diff : 0.0

        # Compute dθ in-place
        inv_2μ2 = 1 / (2 * μ2)
        @. dθ_temp = (-λ + μ1 * diff_θ) * inv_2μ2

        # Normalize dθ
        dθ_norm = sqrt(sum(abs2, dθ_temp))
        @. dθ_temp /= dθ_norm

        # Update costate in-place
        @. λ_out = -2 * μ2 * dθ_temp

        return integ
    end
    return reset_costate!
end

"""
    build_affect(c::MDCProblem, ::StateAffect)

resets state so that residual is zero. also resets costate necessarily. NOT YET FULLY IMPLEMENTED
min C(θ) such that norm(θ - θ₀)^2 = K where K is current distance
we will do this with unconstrained optimisation and lagrange multipliers
ideally would have an inequality constraint >=K. But Optim.jl doesn't support this
"""
function build_affect(c::CurveProblem, ::StateAffect, ::MDCDynamics)
    N = num_params(c)
    H = c.momentum
    cost = c.cost
    θ₀ = initial_params(c)
    dp = param_template(c)
    _reset_costate! = build_affect(c, CostateAffect())
    function reset_state!(integ, dθ)
        K = sum((integ.u[1:N] - θ₀) .^ 2)
        println(K)
        function constr(x) # constraint func: g = 0
            return K - sum((x - θ₀) .^ 2)
        end

        function L(x)
            θ, λ = x[1:(end - 1)], x[end]
            return cost(θ) + λ * constr(θ)
        end
        gc = deepcopy(θ₀)
        function L(x, g)
            θ, λ = x[1:(end - 1)], x[end]
            C = cost(θ, dθ) # dθ is just an arbitrary pre-allocation
            cstr = constr(θ)
            g[1:(end - 1)] = gc + 2 * λ * (θ - θ₀)
            g[end] = cstr
            return C + λ * cstr
        end
        g = zeros(N + 1)
        opt = optimize(L, cat(integ.u[1:N], 0.0, dims = 1), LBFGS())
        C0 = cost(θ₀)
        @info "cost after readjustment is $(opt.minimum). cost before readjustment was $C0"
        (opt.ls_success == true) && (integ.u[1:N] = opt.minimizer[1:N])
        integ = _reset_costate!(integ, dθ)
        return integ
    end
    return integ -> reset_state!(integ, dp)
end

function saving_callback(prob::ODEProblem, saved_values::SavedValues)
    # save states in case simulation is interrupted
    saving_cb = SavingCallback(
        (u, t, integrator) -> u[1:(length(u) ÷ 2)],
        saved_values, saveat = 0.0:0.1:prob.tspan[end]
    )
    return remake(prob, callback = saving_cb)
end

function build_callbacks(c::CurveProblem, callbacks::SciMLBase.DECallback)
    # DECallback supertype includes CallbackSet
    return CallbackSet(callbacks)
end

build_callbacks(c::CurveProblem, n::Nothing) = nothing

function build_callbacks(c::CurveProblem, mdc_callbacks::Vector{T}, mtol::Number) where {
        T <:
        CallbackCallable,
    }
    if !any(x -> x isa MomentumReadjustment, mdc_callbacks)
        push!(mdc_callbacks, MomentumReadjustment(mtol))
    end
    push!(mdc_callbacks, TerminalCond())
    actual_callbacks = map(mdc_callbacks) do cb
        a = cb(c)
    end |> x -> vcat(x...)
    return actual_callbacks
end

function Base.summary(io::IO, prob::MDCProblem)
    type_color, no_color = SciMLBase.get_colorizers(io)
    return print(
        io,
        type_color, nameof(typeof(prob)),
        no_color, " with uType ",
        type_color, typeof(prob.p0),
        no_color, " and tType ",
        type_color,
        prob.tspan isa Function ?
            "Unknown" : (
                prob.tspan === nothing ?
                "Nothing" : typeof(prob.tspan[1])
            ),
        no_color,
        " holding cost function of type ", type_color, nameof(typeof(prob.cost)), no_color
    )
end

function Base.show(io::IO, mime::MIME"text/plain", A::MDCProblem)
    type_color, no_color = SciMLBase.get_colorizers(io)
    summary(io, A)
    println(io)
    print(io, "timespan: ", A.tspan, "\n")
    print(io, "momentum: ", A.momentum, "\n")
    print(io, "Initial parameters p0: ", A.p0, "\n")
    return print(io, "Initial parameter direction dp0: ", A.dp0, "\n")
end
