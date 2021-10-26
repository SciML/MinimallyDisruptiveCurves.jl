"""
    evolve(c::CurveProblem, solmethod=Tsit5; mdc_callback=nothing, callback=nothing, momentum_tol=1e-3,kwargs...)
Evolves a minimally disruptive curve, with curve parameters specified by curveProblem. Uses DifferentialEquations.solve() to run the ODE.
MinimallyDisruptiveCurves.jl callbacks go in the `mdc_callback` keyword argument. You can also use any DifferentialEquations.jl callbacks compatible with DifferentialEquations.solve(). They go in the `callback` keyword argument
"""
function evolve(c::CurveProblem, solmethod=Tsit5; mdc_callback=CallbackCallable[], callback=nothing, momentum_tol=1e-3,kwargs...) 
    
    (!(eltype(mdc_callback) == CallbackCallable)) && (mdc_callback = convert(Vector{CallbackCallable}, mdc_callback))

    function merge_sols(neg, pos, p)
        t = cat(neg.t, pos.t, dims=1)
        u = cat(neg.u, pos.u, dims=1)
        return DiffEqBase.build_solution(p, solmethod, t, u)
    end
    
    probs = c()
    

    callbacks = CallbackSet(
        build_callbacks(c, mdc_callback, momentum_tol)...,
        build_callbacks(c, callback)
    )

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
end
