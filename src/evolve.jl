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




function evolve(c::curveProblem, solmethod=nothing; callback=nothing, kwargs...) 
    ## only worry about positive span, or two sided. forget negative
    p = make_ODEProblem(c)
    callback = CallbackSet(callback, TerminalCond(c.cost,c.momentum) )
    println("lo")
    function merge_sols(furst, second)
        t = cat(furst.t, second.t, dims=1)
        u = cat(furst.u, second.u, dims=1)
        sol = DiffEqBase.build_solution(p, solmethod, t, u)
    end

    solmethod === nothing && (solmethod == Tsit5)
    span = p.tspan
    runn(p) = solve(p, solmethod(); callback=callback, kwargs...)
    
    if span[1] < 0.
        spans = [(span[1], 0.), (0., span[2])]
        pp = remake(p; tspan = spans[2])
        np = remake(p; tspan = spans[1], u0 = -pp.u0)      
        psol = Threads.@spawn runn(pp)
        nsol = runn(np)
        nsol.u[:] = nsol.u[end:-1:1]
        wait(psol)
        psol = psol.result
        sol = merge_sols(nsol, psol)
    else
        sol = runn(p)
    end
    return MinimallyDisruptiveCurve(sol)
end
    
    

