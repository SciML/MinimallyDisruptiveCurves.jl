using OrdinaryDiffEq
# using Plots
using DiffEqParamEstim
using MinimallyDisruptiveCurves
using LinearAlgebra
# gr()


    
# include("../models/forced_mass_spring.jl")
# include("models/circadian_model2.jl")
include("../models/NFKB.jl")
# include("models/STG_Liu.jl")
od,ic,tspan,ps = create()
prob = ODEProblem(od,ic,tspan,ps)

tsteps = range(0, stop=tspan[end], length=100)
nomsol = solve(prob,Tsit5(), saveat=tsteps)
data = hcat(nomsol(tsteps)...)


l2l(sol) = sum([sum(abs2, el) for el in (output_map.((sol(tsteps).u)) .- output_map.((nomsol(tsteps).u)))])
L2L = L2Loss(tsteps, data)
cost = build_loss_objective(prob, Tsit5(), l2l; mpg_autodiff=true)

p0 = prob.p
hess = FiniteDiff.finite_difference_hessian(cost, p0)
# hess = l2_hessian(nomsol)

F = eigen(hess)
dθ₀ = F.vectors[:,1]


mom = 2.
eprob = curveProblem(cost, p0, dθ₀; momentum=mom, span= (0.,1.))
cb1 = MomentumReadjustment(cost, eprob; momentum = mom, tol = 1e-3)
cb2 = VerboseOutput(:low, 0.1:0.1:10)
# cb3 = ParameterBounds([1,2,3], [0.,0.,0.], [10.,10.,10.])
cb = CallbackSet(cb1,cb2)
esol = solve(eprob, Tsit5(); callback=cb, maxiter=300)

