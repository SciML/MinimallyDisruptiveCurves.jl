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




# https://discourse.julialang.org/t/du-i-not-the-same-as-integrator-t-val-1-differentialequations-jl/39616 
# for info on saving callbacks (ie res)
function curveProblem(cost, θ₀, dθ₀; momentum=10., span=(-3.,3.)) 
    N = length(θ₀)
    ∇C = copy(dθ₀) # pre-assignment for mutation
    λ₀ = initial_costate(dθ₀, momentum, cost(θ₀))
    u₀ = cat(θ₀,λ₀,dims=1)
    f = (du,u,p,t) -> evolveODE(du,u,p,t, cost, ∇C, N, momentum, θ₀)
    evoProb = ODEProblem(f, u₀, span)
   return  evoProb
end


function evolveODE(du ,u , p, t, cost, ∇C, N, H, θ₀)
 
    θ = u[1:N] # current parameter vector
    λ = u[N+1:end] #current costate vector

    dist = sum((θ - θ₀).^2) # should = t actually? check and replace?
    C = cost(θ, ∇C) #also updates ∇C as a mutable

    μ2 = (C-H)/2
    μ1 = dist > 1e-3 ?  (λ'*λ - 4*μ2^2 )/(λ'*(θ - θ₀)) : 0 
        # if mu1 < -1e-4 warn of numerical issue
        # if mu1 > 1e-3 and dist > 1e-3 then set mu1 = 0
    du[1:N] = @. (-λ + μ1*(θ - θ₀))/(2*μ2) # ie dθ
    du[1:N] /= (sqrt(sum((du[1:N]).^2)))
    damping_constant = (λ'*du[1:N])/(H-C)  #theoretically = 1 but not numerically
    du[N+1:end] = @. (μ1*du[1:N] - ∇C)*damping_constant # ie dλ
    res = λ  + 2*μ2*du[1:N]
    # println("μ1K = $(μ1*du[1:N]'*(θ - θ₀))")
    # println("resid is", norm(res))
    # println(t)
    # print_warnings(u,du,N)
    return nothing
end



function evolve(p::P, solmethod=nothing; kwargs...) where P <: ODEProblem
    ## only worry about positive span, or two sided. forget negative
    
    function merge_sols(furst, second)
        t = cat(furst.t, second.t, dims=1)
        u = cat(furst.u, second.u, dims=1)
        sol = DiffEqBase.build_solution(p, solmethod, t, u)
    end
    
    solmethod === nothing && (solmethod == Tsit5())
    span = p.tspan
    runn(p) = solve(p, solmethod(); kwargs...)
    
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
    
    

