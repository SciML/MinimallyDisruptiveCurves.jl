abstract type CurveProblem end


"""
For callbacks to tune MD Curve
"""
abstract type ConditionType end
struct ResidualCondition <: ConditionType end
struct CostCondition <: ConditionType end 

abstract type AffectType end
struct StateAffect <: AffectType end
struct CostateAffect <: AffectType end

abstract type CurveInfoSnippet end
struct EmptyInfo <: CurveInfoSnippet end
struct CurveDistance <: CurveInfoSnippet end
struct HamiltonianResidual <: CurveInfoSnippet end

struct Verbose{T <: CurveInfoSnippet,S <: Real,V <: AbstractRange} 
    snippets::Vector{T}
    timepoints::Union{V{S},Vector{S}}
end

Verbose() = Verbose([EmptyInfo()], 0:0)
Verbose(snippet::EmptyInfo, times) = Verbose()
Verbose(snippet <: CurveInfoSnippet, times) = Verbose([snippet], times)

function (c::CurveDistance)(integ)
    @info "curve length is $(integ.t)"
    nothing
end

function (h::HamiltonianResidual)(integ)
    @info "dHdu residual ="
end

function (v::Verbose)
    function affect!(integ)

        return integ
    end
    return PresetTimeCallback(v.times, affect!)
end


struct MDCProblem{A,B,C,D,E} <: CurveProblem
    cost::A
    p0::B
    dp0::C
    momentum::D 
    tspan::E
    ## reverse initial direction and signflip curve span if the latter is nonpositive
    function MDCProblem(a::A, b::B, c::C, d::D, e::E) where A where B where C where D where E
        if max(e...) <= 0.
            e = map(x -> -x |> abs, e) |> reverse
            c = -c
        end
        new{A,B,C,D,E}(a, b, c, d, e)
    end
end

num_params(c::CurveProblem) = length(c.p0)
param_template(c::CurveProblem) = deepcopy(c.p0)
initial_params(c::CurveProblem) = c.p0

function curveProblem(a, b, c, d, e)
    @warn("curveProblem and specify_curve are DEPRECATED. please use MDCProblem (with the same arguments) instead")
    return MDCProblem(a, b, c, d, e)
end

specify_curve(cost, p0, dp0, momentum, tspan) = curveProblem(cost, p0, dp0, momentum, tspan)
specify_curve(;cost=nothing, p0=nothing, dp0=nothing,momentum=nothing,tspan=nothing) = curveProblem(cost, p0, dp0, momentum, tspan)



"""
    (c::MDCProblem)()
returns a tuple of ODEProblems specificed by the MDCProblem. Usually a single ODEProblem. Two are provided if the curve crosses zero, so that one can run two curves in parallel going backwards/forwards from zero
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
function make_spans(c::MDCProblem, span)
    (span[1] > span[2]) && error("make your curve span monotone increasing")
    if (span[2] > 0) && (span[1] < 0)
        spans = ((0., span[1]), (0., span[2]))
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
    μ₂ = (-c.momentum + c.cost(c.p0)) / 2.
    λ₀ = -2. * μ₂ * c.dp0 
    return λ₀
end

"""
Generate initial conditions of an MDCProblem
"""
function initial_conditions(c::MDCProblem)
    λ₀ = initial_costate(c)
    return cat(c.p0, λ₀, dims=1)
end

"""
Generate vector field for MD curve, as specified by MDCProblem
"""
function dynamics(c::MDCProblem)
    cost = c.cost
    ∇C = param_template(c)
    N = num_params(c)
    H = c.momentum
    θ₀ = initial_params(c)
    function upd(du, u, p, t)
        θ = u[1:N] # current parameter vector
        λ = u[N + 1:end] # current costate vector
        dist = sum((θ - θ₀).^2) # which should = t so investigate cost/benefits of using t instead of dist
        C = cost(θ, ∇C) # also updates ∇C as a mutable
        μ2 = (C - H) / 2
        μ1 = dist > 1e-3 ?  (λ' * λ - 4 * μ2^2 ) / (λ' * (θ - θ₀)) : 0. 
        # if mu1 < -1e-4 warn of numerical issue
        # if mu1 > 1e-3 and dist > 1e-3 then set mu1 = 0
        du[1:N] = @. (-λ + μ1 * (θ - θ₀)) / (2 * μ2) # ie dθ
        du[1:N] /= (sqrt(sum((du[1:N]).^2)))
        damping_constant = (λ' * du[1:N]) / (H - C)  # theoretically = 1 but not numerically
        du[N + 1:end] = @. (μ1 * du[1:N] - ∇C) * damping_constant # ie dλ
    res = λ  + 2 * μ2 * du[1:N]
        return nothing
    end
    return upd
end

"""
Callback to stop MD Curve evolving if cost > momentum
"""
function TerminalCond(c::MDCProblem)
    cost = c.cost
    H = c.momentum
N = num_params(c)
    function condition(u, t, integrator)
        return (cost(u[1:N]) > H)
    end
    return DiscreteCallback(condition, terminate!)
end
    
"""
    Callback to readjust momentum in the case that the numerical residual from the identity dHdu = 0 crosses a user-specified threshold
"""
MomentumReadjustment(c::CurveProblem, tol; kwargs...) = readjustment(c, ResidualCondition(), CostateAffect(), tol; kwargs...)

"""
    Callback to readjust state in the case that the numerical residual from the identity dHdu = 0 crosses a user-specified threshold. EXPERIMENTAL AND WILL PROBABLY BREAK
"""
StateReadjustment(c::CurveProblem, tol; kwargs...) = readjustment(c, ResidualCondition(), StateAffect(), tol; kwargs...)
        
    
function readjustment(c::CurveProblem, cnd::ConditionType, aff::AffectType, momentum_tol; kwargs...)
    if isnan(momentum_tol)
        return nothing
    end
    cond = build_cond(c, cnd, momentum_tol)
    affect! = build_affect(c, aff)
    return DiscreteCallback(cond, affect!)
end

function build_cond(c::MDCProblem, ::ResidualCondition, tol)
    N = num_params(c)
    H = c.momentum
    θ₀ = initial_params(c)
    dθ = param_template(c)

    function rescond(u, t, integ)
        absres = dHdu_residual(c, u, t, dθ) 
        absres > tol ? begin
            # @info "applying readjustment at t=$t, |res| = $absres"
            return true
        end : return false
    end
    return rescond
end
    
function build_cond(c::MDCProblem, ::CostCondition, tol)
    N = num_params(c) 
    function costcond(u, t, integ)
         (c.cost(u[1:N]) > tol) ? (return true) : (return false)
     end
     return costcond
end

"""
    I wanted to put dθ[:] = ... here instead of dθ = ... . Somehow the output of the MDC changes each time if I do that, there is a dirty state being transmitted. But I don't at all see how from the code. Figure out.
"""


"""
Checks dHdu residual (u deriv of Hamiltonian). Returns true if residual is greater than some tolerance (it should be zero)
"""
    function dHdu_residual(c::MDCProblem, u, t, dθ)
    N = num_params(c)
    H = c.momentum
    θ₀ = initial_params(c)
    
    θ = u[1:N] 
    λ = u[N + 1:end]
    μ2 = (c.cost(θ) - H) / 2.
μ1 = t > 1e-3 ?  (λ' * λ - 4 * μ2^2 ) / (λ' * (θ - θ₀)) : 0.
    dθ = (-λ + μ1 * (θ - θ₀)) / (2 * μ2)
    dθ /= (sqrt(sum((dθ).^2))) 
    return sum(abs.(λ + 2 * μ2 * dθ))
end


"""
    build_affect(c::MDCProblem, ::CostateAffect)
Resets costate to undo effect of cumulative numerical error. Specifically, finds costate so that dHdu = 0, where H is the Hamiltonian.

*I wanted to put dθ[:] = ... here instead of dθ = ... . Somehow the output of the MDC changes each time if I do that, there is a dirty state being transmitted. But I don't at all see how from the code. Figure out.*
"""
function build_affect(c::MDCProblem, ::CostateAffect)
    N = num_params(c)
    H = c.momentum
    θ₀ = initial_params(c)
    dp = param_template(c)
    function reset_costate!(integ, dθ)
        θ = integ.u[1:N] # current parameter vector
        λ = integ.u[N + 1:end] # current costate vector 
        μ2 = (c.cost(θ) - H) / 2
        μ1 = integ.t > 1e-3 ?  (λ' * λ - 4 * μ2^2 ) / (λ' * (θ - θ₀)) : 0. 
        dθ = (-λ + μ1 * (θ - θ₀)) / (2 * μ2) 
        dθ /= (sqrt(sum((dθ).^2)))
    integ.u[N + 1:end] =  -2 * μ2 * dθ
        return integ
    end
    return integ -> reset_costate!(integ, dp)
end


"""
    build_affect(c::MDCProblem, ::StateAffect)
resets state so that residual is zero. also resets costate necessarily. NOT YET FULLY IMPLEMENTED
min C(θ) such that norm(θ - θ₀)^2 = K where K is current distance
we will do this with unconstrained optimisation and lagrange multipliers
ideally would have an inequality constraint >=K. But Optim.jl doesn't support this
"""
function build_affect(c::MDCProblem, ::StateAffect)
        N = num_params(c)
    H = c.momentum
    cost = c.cost
        θ₀ = initial_params(c)
    dp = param_template(c)
    _reset_costate! = build_affect(c, CostateAffect())
    function reset_state!(integ, dθ)
        K = sum((integ.u[1:N] - θ₀).^2)
        println(K)
        function constr(x) # constraint func: g = 0
            return K - sum((x - θ₀).^2)
        end
    
        function L(x)
            θ, λ = x[1:end - 1], x[end]
            return cost(θ) + λ * constr(θ)
        end
        gc = deepcopy(θ₀)
        function L(x, g)
            θ, λ = x[1:end - 1], x[end]
            C = cost(θ, dθ) # dθ is just an arbitrary pre-allocation
            cstr = constr(θ)
            g[1:end - 1] = gc + 2 * λ * (θ - θ₀)
            g[end] = cstr
            return C + λ * cstr
            return 
        end
        g = zeros(N + 1)
        # lagr = L(cat(θ₀.+1., 0., dims=1), g)
        # println("new gradient is $g")
        # println("Lagrangian is $lagr")
        # println("θ is ")
        opt = optimize(L, cat(integ.u[1:N], 0., dims=1), LBFGS())
        C0 = cost(θ₀)
        @info "cost after readjustment is $(opt.minimum). cost before readjustment was $C0"
        (opt.ls_success == true) && (integ.u[1:N] = opt.minimizer[1:N])
        integ = _reset_costate!(integ, dθ)
        return integ
    end
    return integ -> reset_state!(integ, dp)
end

function build_callbacks(c::MDCProblem, callbacks, momentum_tol, kwargs...)
    return CallbackSet(callbacks, TerminalCond(c), MomentumReadjustment(c, momentum_tol; kwargs...))
end

