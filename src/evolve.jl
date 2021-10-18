"""
    evolve(c::curveProblem, solmethod=nothing; callback=nothing, momentum_tol = 1e-3,kwargs...)
Evolves a minimally disruptive curve, with curve parameters specified by curveProblem. Uses DifferentialEquations.solve() to run the ODE.
As well as MinimallyDisruptiveCurves.jl callbacks, you can use any DifferentialEquations.jl callbacks compatible with DifferentialEquations.solve(). 
"""
function evolve(c::CurveProblem, solmethod=Tsit5; callback=nothing, momentum_tol=1e-3,kwargs...) 
    
    function merge_sols(neg, pos, p)
        t = cat(neg.t, pos.t, dims=1)
        u = cat(neg.u, pos.u, dims=1)
        return DiffEqBase.build_solution(p, solmethod, t, u)
    end
    
    probs = c()
    callbacks = build_callbacks(c, callback, momentum_tol, kwargs...)
    # sols = map(probs) do prob
    #     solve(prob, solmethod(); callback=callbacks, kwargs...)
    # end
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
    return MDCSolution(sols, c.cost)
    # return map(sol -> MinimallyDisruptiveCurve(sol, c.cost), sols)
end
