"""
    evolve(c::CurveProblem, solmethod=Tsit5; mdc_callback=nothing, callback=nothing, saved_values=nothing, momentum_tol=1e-3,kwargs...)

Evolves a minimally disruptive curve, with curve parameters specified by curveProblem. Uses DifferentialEquations.solve() to run the ODE.
MinimallyDisruptiveCurves.jl callbacks go in the `mdc_callback` keyword argument. You can also use any DifferentialEquations.jl callbacks compatible with DifferentialEquations.solve(). They go in the `callback` keyword argument
"""
function evolve(
        c::CurveProblem, solmethod = Tsit5; mdc_callback = CallbackCallable[],
        callback = nothing, saved_values = nothing, momentum_tol = 1.0e-3, kwargs...
    )
    (!(eltype(mdc_callback) == CallbackCallable)) &&
        (mdc_callback = convert(Vector{CallbackCallable}, mdc_callback))

    function merge_sols(ens, p)
        if length(ens.u) == 1
            return ens.u[1]
        elseif length(ens.u) == 2
            t = cat(-ens.u[1].t[end:-1:1], ens.u[2].t, dims = 1)
            u = cat(ens.u[1].u[end:-1:1], ens.u[2].u, dims = 1)
            return SciMLBase.build_solution(p, Tsit5(), t, u)
        end
    end

    probs = c()
    !isnothing(saved_values) && (probs = saving_callback.(probs, saved_values))

    # EnsembleProblem's prob_func signature differs across SciMLBase versions:
    # v2 calls `(prob, i::Int, repeat)`, v3 calls `(prob, ctx::EnsembleContext)`
    # where the trajectory index is `ctx.sim_id`.
    prob_func(prob, i::Integer, repeat) = probs[i]
    prob_func(prob, ctx) = probs[ctx.sim_id]

    callbacks = CallbackSet(
        build_callbacks(c, mdc_callback, momentum_tol)...,
        build_callbacks(c, callback)
    )

    e = EnsembleProblem(probs[1], prob_func = prob_func)
    sim = solve(
        e, OrdinaryDiffEq.Tsit5(), EnsembleThreads(), trajectories = length(probs);
        callback = callbacks, kwargs...
    )
    return merge_sols(sim, probs[1]) |> MDCSolution
end

function evolvenew(
        c::CurveProblem; mdc_callback = CallbackCallable[],
        callback = nothing, saved_values = nothing, momentum_tol = 1.0e-3, kwargs...
    )

 (!(eltype(mdc_callback) == CallbackCallable)) &&
        (mdc_callback = convert(Vector{CallbackCallable}, mdc_callback))

    end
