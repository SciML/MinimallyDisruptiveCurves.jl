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

