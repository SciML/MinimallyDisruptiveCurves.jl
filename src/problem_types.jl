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

struct MDCProblem{A,B,C,D,E} <: CurveProblem
    cost::A
    p0::B
    dp0::C
    momentum::D 
    tspan::E
    ## reverse initial direction and signflip curve span if the latter is nonpositive
    # function MDCProblem(a::A, b::B, c::C, d::D, e::E) where A where B where C where D where E
    #     if max(e...) <= 0.
    #         e = map(x -> -x |> abs, e)
    #         c = -c
    #         println("hi")
    #     end
    #     new{A,B,C,D,E}(a, b, c, d, e)
    # end
end

num_params(c::CurveProblem) = length(c.p0)
param_template(c::CurveProblem) = deepcopy(c.p0)
initial_params(c::CurveProblem) = c.p0

# does equivalent of make_ODEProblem
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

function make_spans(c::MDCProblem, span)
    (span[1] > span[2]) && error("make your curve span monotone increasing")
    if (span[2] > 0) && (span[1] < 0)
        spans = ((0., span[1]), (0., span[2]))
    else
        spans = (span,)
    end
    return spans
end

function initial_costate(c::MDCProblem)
    μ₂ = (-c.momentum + c.cost(c.p0)) / 2.
    λ₀ = -2. * μ₂ * c.dp0 
    return λ₀
end

function initial_conditions(c::MDCProblem)
    λ₀ = initial_costate(c)
    return cat(c.p0, λ₀, dims=1)
end

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

function TerminalCond(c::MDCProblem)
    cost = c.cost
    H = c.momentum
    N = num_params(c)
    function condition(u, t, integrator)
        return (cost(u[1:N]) > H)
    end
    return DiscreteCallback(condition, terminate!)
end
    
MomentumReadjustment(c::CurveProblem, tol; kwargs...) = readjustment(c, ResidualCondition(), CostateAffect(), tol; kwargs...)

StateReadjustment(c::CurveProblem, tol; kwargs...) = readjustment(c, ResidualCondition(), StateAffect(), tol; kwargs...)
        
    
function readjustment(c::CurveProblem, cnd::ConditionType, aff::AffectType, momentum_tol; kwargs...)
    if isnan(momentum_tol)
        return nothing
    end
    cond = build_cond(c, cnd, momentum_tol)
    affect! = build_affect(c, aff)
    cb = DiscreteCallback(cond, affect!)
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
    I wanted to put dθ[:] = ... here instead of dθ = ... . Somehow the output of the MDC changes each time if I do that, there is a dirty state being transmitted. But I don't at all see how from the code. Figure out.
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