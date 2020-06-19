using DiffEqParamEstim
using RecursiveArrayTools
using DifferentialEquations
using Plots
using LinearAlgebra
using Optim
gr()


    
step(t) = max(t,1.)
inputs = [sin,cos,step]
include("models/forced_mass_spring.jl")

alls = create.(inputs)
tspan = (0.,100.)

probs = [ODEProblem(el[1],el[2],tspan, el[3]) for el in alls]
ic = alls[1][2]
ps = alls[1][3]
tstrct = logabs_transform(last.(ps))
probs = [transform_problem(prob,tstrct; unames = first.(ic), pnames = first.(ps)) for prob in probs]
probs = last.(probs)
nomsols = [solve(prob, Tsit5()) for prob in probs]
l2l(sol, nominal) = sum((sol(t) - nominal(t)).^2)
losses = [sol -> l2l(sol, nomsol) for nomsol in nomsols]
costs = [build_loss_objective(prob,Tsit5(),each_loss; mpg_autodiff=true) for (prob,each_loss) in zip(probs,losses)]


t = collect(range(0,stop=10,length=100) )
l2l(sol, nominal) = sum((sol(t) - nominal(t)).^2)

losses = [sol -> l2l(sol, nomsol) for nomsol in nominal_sols]
costs = [build_loss_objective(prob,Tsit5(),each_loss; mpg_autodiff=true) for each_loss in losses]


# # cost_function = build_loss_objective(prob,Tsit5(),l2l)
# cost_function = build_loss_objective(prob,Tsit5(),l2l; mpg_autodiff=true)





# o = optimize(cost_function, p0; g_abstol=1e-9)
# popt = o.minimizer

# # function showit()
# #     plot(sol)
# #     scatter!(t,data)
# # end

# dθ₀ = -initial_md_direction(cost_function, popt)
# eprob, esol = evolve(cost_function, popt, dθ₀)