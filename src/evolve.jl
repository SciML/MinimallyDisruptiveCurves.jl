"""
Code for evolving a minimally disruptive curve

Needs:
momentum (= final cost)
initial parameter
initial direction
normal solve kwargs:
    maxiters
    algorithm eg Tsit5()                
"""

# function specify_curve(cost, θ₀, dθ₀; momentum=10., span=(-3.,3.)) 
#     N = length(θ₀)
#     ∇C = copy(dθ₀) # pre-assignment for mutation
#     λ₀ = initial_costate(dθ₀, momentum, cost(θ₀))
#     u₀ = cat(θ₀,λ₀,dims=1)
#     f = (du,u,p,t) -> evolveODE(du,u,p,t, cost, ∇C, N, momentum, θ₀)
#     ode = ODEProblem(f, u₀, span)
#    return curveProblem(ode, cost, )
# end



"""
    evolve(c::curveProblem, solmethod=nothing; callback=nothing, momentum_tol = 1e-3,kwargs...)
Evolves a minimally disruptive curve, with curve parameters specified by curveProblem. Uses DifferentialEquations.solve() to run the ODE.
As well as MinimallyDisruptiveCurves.jl callbacks, you can use any DifferentialEquations.jl callbacks compatible with DifferentialEquations.solve(). 
"""
function evolve(c::curveProblem, solmethod=nothing; callback=nothing, momentum_tol = 1e-3,kwargs...) 
    ## only worry about positive span, or two sided. forget negative
    p = make_ODEProblem(c)
    if isnan(momentum_tol) == false
        momc = MomentumReadjustment(c.cost, c.p0; momentum = c.momentum, tol = momentum_tol)
    else
        momc = nothing
    end
        callback = CallbackSet(callback, TerminalCond(c.cost,c.momentum), momc)

    function merge_sols(furst, second)
        t = cat(furst.t, second.t, dims=1)
        u = cat(furst.u, second.u, dims=1)
        sol = DiffEqBase.build_solution(p, solmethod, t, u)
    end

    solmethod === nothing && (solmethod == Tsit5)
    span = p.tspan
    runn(x) = solve(x, solmethod(); callback=callback, kwargs...)
    
    if span[1] < 0.
        spans = [(0., -span[1]), (0., span[2])]
        cplus = c
        cplus.tspan = spans[2]
        pp = make_ODEProblem(cplus)
        
        cminus = c
        cminus.tspan = spans[1]
        cminus.dp0 = -cplus.dp0
        np = make_ODEProblem(cminus)     
        psol = Threads.@spawn runn(pp)
        nsol = runn(np)
        nsol.u[:] = nsol.u[end:-1:1]
        nsol.t[:] = -nsol.t[end:-1:1]
        wait(psol)
        psol = psol.result
        sol = merge_sols(nsol, psol)
    else
        sol = runn(p)
    end
    return MinimallyDisruptiveCurve(sol, c.cost)
end
    
    

MinimallyDisruptiveCurve