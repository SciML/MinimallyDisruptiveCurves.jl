using OrdinaryDiffEq
# using Plots
using MinimallyDisruptiveCurves
using DiffEqParamEstim
# gr()


heaviside = soft_heaviside(0.01, 3600)
inputs = [heaviside]


# include("../models/forced_mass_spring.jl")
# include("../models/circadian_model2.jl")

include("../models/NFKB.jl")
# include("../models/STG_Liu.jl")


od, ic, ps = create()
tspan = (0., 100.)
p0 = last.(ps)
prob = ODEProblem(od,ic,tspan,ps)
ts = 0.:0.5:100.
cost = build_injection_loss(prob, Tsit5(), ts)
# cost = build_loss_objective(lp, Tsit5(), ls; mpg_autodiff=true)

dθ₀ = initial_md_direction(cost, p0)
mom = 10.
eprob = curveProblem(cost, p0, dθ₀; momentum=mom, span= (0.,50.))
cb1 = MomentumReadjustment(cost, eprob; momentum = mom, tol = 1e-3)
cb2 = VerboseOutput(:low, 0.1:0.1:10)
# cb3 = ParameterBounds([1,2,3], [0.1,0.1,0.1], [10.,10.,10.])
cb = CallbackSet(cb1,cb2)
esol = solve(eprob, Tsit5(); callback=cb, maxiter=3000)

