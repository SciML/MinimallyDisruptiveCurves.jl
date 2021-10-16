"""
Code for evolving a minimally disruptive curve                
"""


"""
    evolve(c::curveProblem, solmethod=nothing; callback=nothing, momentum_tol = 1e-3,kwargs...)
Evolves a minimally disruptive curve, with curve parameters specified by curveProblem. Uses DifferentialEquations.solve() to run the ODE.
As well as MinimallyDisruptiveCurves.jl callbacks, you can use any DifferentialEquations.jl callbacks compatible with DifferentialEquations.solve(). 
"""
function evolve(c::curveProblem, solmethod=nothing; callback=nothing, momentum_tol=1e-3,kwargs...) 
    ## only worry about positive span, or two sided. forget negative
    p = make_ODEProblem(c)
    if isnan(momentum_tol) == false
        momc = MomentumReadjustment(c.cost, c.p0; momentum=c.momentum, tol=momentum_tol)
    else
        momc = nothing
    end
    callback = CallbackSet(callback, TerminalCond(c.cost, c.momentum), momc)
    # println(momentum_tol, "for old")
    # println("terminal condition momentum", c.momentum)
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


function evolve(c::CurveProblem, solmethod=Tsit5; callbacks=nothing, momentum_tol=1e-3,kwargs...) 
    
    function merge_sols(neg, pos, p)
        t = cat(neg.t, pos.t, dims=1)
        u = cat(neg.u, pos.u, dims=1)
        return DiffEqBase.build_solution(p, solmethod, t, u)
    end
    
    probs = c()
    callbacks = build_callbacks(c, callbacks, momentum_tol, kwargs...)
    println(momentum_tol, "for new")
    println("terminal condition momentum", c.momentum) 
    # sols = map(probs) do prob
    #     solve(prob, solmethod(); callback=callbacks, kwargs...)
    # end
    println(momentum_tol)

    runn(p) = solve(p, solmethod(); callback=callbacks, kwargs...)

    (length(probs) == 1) && (sols = runn(probs[1]))

    if length(probs) == 2
        psol = Threads.@spawn runn(probs[2])
        nsol = runn(probs[1])
        nsol.u[:] = nsol.u[end:-1:1]
        nsol.t[:] = -nsol.t[end:-1:1]
        wait(psol)
        psol = psol.result
        sols = merge_sols(nsol, psol, probs[end])
    end
    println(c.p0)
    return MinimallyDisruptiveCurve(sols, c.cost)
    # return map(sol -> MinimallyDisruptiveCurve(sol, c.cost), sols)
end
