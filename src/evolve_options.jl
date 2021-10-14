abstract type Readjustment end
struct Momentum end <: Readjustment
struct State end <: Readjustment

abstract type ReadjustmentCondition end
struct Residual end <: ReadjustmentCondition

"""
    readjustment(cost, θ₀, readjust; momentum = 10., condition = :res, tol = 1e-3)
condition = :res or :cost 
:res means |dHdu| > tol for momentum readjustment to occur
:cost means C(θ) > tol for momentum readjustment to occur
"""
function readjustment(cost, θ₀, readjust; momentum=10., condition=:res, tol=1e-3)
    
    N = length(θ₀)
    dθ = deepcopy(θ₀)
    if condition == :res
        cond = (u, t, integ) -> rescond(u,t,integ, 
                                        momentum, 
                                        cost, 
                                        θ₀,
                                        dθ, 
                                        tol)    
    elseif condition == :cost
        cond = (u, t, integ) -> costcond(u, t, integ, cost, tol)
    end
    costate_affect! = integ -> reset_costate!(integ, N, momentum, cost, dθ, θ₀)
    state_affect! = integ -> reset_state!(integ, N, momentum, cost,  dθ, θ₀)
    readjust == :state && (affect! = state_affect!)
    readjust == :momentum && (affect! = costate_affect!)
    cb = DiscreteCallback(cond, affect!)
    return cb
end

MomentumReadjustment(cost, prob; kwargs...) = readjustment(cost, prob, :momentum; kwargs...)
StateReadjustment(cost, prob; kwargs...) = readjustment(cost, prob, :state; kwargs...)

MomentumReadjustment(c::MDCProblem)


"""
Checks dHdu residual (u deriv of Hamiltonian). Returns true if residual is greater than some tolerance (it should be zero)
"""
function rescond(u, t, integrator, H, cost, θ₀, dθ, tol)
    absres = calculate_dHdu_residual(u, t, H, cost, θ₀, dθ)
    if absres > tol
    @info "applying readjustment at t=$t, |res| = $absres"
    return true
    else
        return false
end
end


"""
    calculate_dHdu_residual(u,t,H,cost, θ₀, dθ)
u = cat(state, costate)
t = time (curve distance)
dθ = proposed d/dt(state)
"""
function calculate_dHdu_residual(u, t, H, cost, θ₀, dθ)
    N = length(u) ÷ 2
    θ = u[1:N] # current parameter vector
    λ = u[N + 1:end] # current costate vector
    μ2 = (cost(θ) - H) / 2
    μ1 = t > 1e-3 ?  (λ' * λ - 4 * μ2^2 ) / (λ' * (θ - θ₀)) : 0
    dθ = @. (-λ + μ1 * (θ - θ₀)) / (2 * μ2)
    dθ /= (sqrt(sum((dθ).^2)))
    return sum(abs.(λ + 2 * μ2 * dθ))
end


"""
    a DiscreteCallback condition
"""
function costcond(u, t, integrator, cost, tol)
    θ = u[1:N] # current parameter vector
    C = cost(θ)
    (C > tol) ? (return true) : (return false)
end


"""
    resets costate so that the residual is zero
"""
function reset_costate!(integ, N, H, cost, dθ, θ₀)
    θ = integ.u[1:N] # current parameter vector
    λ = integ.u[N + 1:end] # current costate vector 
    μ2 = (cost(θ) - H) / 2
    μ1 = integ.t > 1e-3 ?  (λ' * λ - 4 * μ2^2 ) / (λ' * (θ - θ₀)) : 0 
    dθ = @. (-λ + μ1 * (θ - θ₀)) / (2 * μ2)
    dθ /= (sqrt(sum((dθ).^2)))
    integ.u[N + 1:end] =  -2 * μ2 * dθ
end

"""
    resets state so that residual is zero. also resets costate necessarily. NOT YET FULLY IMPLEMENTED
    min C(θ) such that norm(θ - θ₀)^2 = K where K is current distance
    we will do this with unconstrained optimisation and lagrange multipliers
    ideally would have an inequality constraint >=K. But Optim.jl doesn't support this
"""
function reset_state!(integ, N, H, cost, dθ, θ₀)
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
reset_costate!(integ, N, H, cost, dθ, θ₀)
    return integ
end

"""
    VerboseOutput(level=:low, times = 0:0.1:1.)
    Callback to give online info on how the solution is going, as the MDCurve evolves. activates at curve distances specified by times
"""
function VerboseOutput(level=:low, times=0:0.1:1.)
    
    function affect!(integ)
        if level == :low 
            @info "curve length is $(integ.t)"
        end
        if level == :medium 
        @info "dHdu residual = "
    end
    if level == :high

        end
        return integ
end
    return PresetTimeCallback(times, affect!) 
end

"""
    ParameterBounds(ids::Vector{Integer},lbs::Vector{Number},ubs::Vector{Number})
parameters[ids] must fall within lbs and ubs, where lbs and ubs are Arrays of the same size as ids.
Create hard bounds on the parameter space over which the minimally disruptive curve can trace. Curve evolution terminates if it hits a bound.
"""
function ParameterBounds(ids, lbs, ubs)
        function condition(u, t, integrator)
        tests = u[ids]
        any(tests .< lbs) && return true
        any(tests .> ubs) && return true
        return false
    end
    return DiscreteCallback(condition, terminate!)
end


function TerminalCond(cost, H)
    function condition(u, t, integrator)
        N = length(u) ÷ 2
        return (cost(u[1:N]) > H)
    end
    return DiscreteCallback(condition, terminate!)
end

    function build_callbacks(c, callbacks, momentum_tol, kwargs...)
    momc = isnan(momentum_tol) ? nothing : 
    MomentumReadjustment(c.cost, c.p0; momentum=c.momentum, tol=momentum_tol)

    return CallbackSet(callbacks, TerminalCond(c.cost, c.momentum), momc)
end