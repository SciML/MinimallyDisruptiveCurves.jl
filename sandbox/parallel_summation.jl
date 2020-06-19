
"""
Haven't gotr parallel summation with pmap to work yet. problems with loading code to the other workers. leaving for now.
Another option is to just multithread. Although the mutable nature of the cost function is then a problem
"""


using MinimallyDisruptiveCurves
using DiffEqParamEstim
include("../models/forced_mass_spring.jl")


heaviside = soft_heaviside(0.01, 1.)

tspan = (0.,10.)
inputs = [heaviside, sin]
tsteps = 0.:1.:10.


outs = create.(inputs)
ods = [outs[i][1] for i in 1:length(outs)]
ics = [outs[i][2] for i in 1:length(outs)]
ps = [outs[i][3] for i in 1:length(outs)]
p0 = last.(ps[1])

probs = [ODEProblem(od,ic,tspan,ps) for (od,ic,ps) in zip(ods,ics,ps)]
nomsols = [solve(prob, Tsit5()) for prob in probs]
single_loss(sol, nomsol) = sum((sol(tsteps) - nomsol(tsteps)).^2)
l2ls = [sol -> single_loss(sol,nomsol) for nomsol in nomsols]
costs = [build_loss_objective(prob, Tsit5(), l2l, mpg_autodiff=true) for (prob,l2l) in zip(probs, l2ls)]


costs = [build_loss_objective(prob, Tsit5(), l2l, mpg_autodiff=true) for (prob,l2l) in zip(probs, l2ls)]

