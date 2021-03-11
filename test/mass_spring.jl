using ModelingToolkit, OrdinaryDiffEq, DiffEqParamEstim, MinimallyDisruptiveCurves, ForwardDiff, LinearAlgebra
  
  function make_model(input)
  @parameters t
  @parameters k,c,m
  D = Differential(t)
  @variables pos(t) vel(t)

  eqs = [D(pos) ~ vel,
        D(vel) ~ (-1/m)*(c*vel + k*pos - input(t))
  ]

  ps = [k,c,m] .=>  [2.,1.,4.] 
  ics = [pos, vel] .=> [1.,0.]
  od = ODESystem(eqs, t, first.(ics), first.(ps)
                                 , default_u0 = Dict(first.(ics) .=> last.(ics))
                                 , default_p = Dict(first.(ps) .=> last.(ps))
                                 )
  tspan = (0.,100.)
  # prob = ODEProblem(od, ics, tspan, ps)
  return od, ics, tspan, ps
end

od, ics, tspan, ps = make_model(t -> 0.)

"""
take a log transform of the od parameter space in two ways: at the level of the ODESystem and the level of the ODEProblem
"""
p0 = last.(ps)
tr = logabs_transform(p0)
log_od = transform_ODESystem(od, tr)
@test typeof(log_od) == ODESystem

prob1 = ODEProblem(od, 
                        collect(ModelingToolkit.get_default_u0(od)), 
                        tspan, 
                        collect(ModelingToolkit.get_default_p(od)))

log_od2, log_ics2, log_ps2 = transform_problem(prob1, tr; unames = ModelingToolkit.get_states(od), pnames = ModelingToolkit.get_ps(od))


"""
check if the two manners of transforming the ODE system give the same output
"""

@test repr.(ModelingToolkit.get_ps(log_od)) == repr.(ModelingToolkit.get_ps(log_od2))

log_prob1 = ODEProblem(log_od, 
        collect(ModelingToolkit.get_default_u0(log_od)), 
        tspan, 
        collect(ModelingToolkit.get_default_p(log_od)))

log_prob2 = ODEProblem(log_od2, 
        collect(ModelingToolkit.get_default_u0(log_od2)), 
        tspan, 
        collect(ModelingToolkit.get_default_p(log_od2)))


sol1 = solve(log_prob1, Tsit5())
sol2 = solve(log_prob2, Tsit5())

@test sol1[end] == sol2[end]

"""
check if  log transforming the cost function on od gives the same result as an untransformed cost function on log_od
"""
tsteps = tspan[1]:1.:tspan[end]
lossf(sol) = sum( [sum(abs2, el1 - el2) for (el1, el2) in zip(sol(tsteps).u, nom_sol(tsteps).u) ] )

nom_cost = build_loss_objective(prob, Tsit5(), lossf; mpg_autodiff=true)

log_cost = build_loss_objective(log_prob1, Tsit5(), lossf; mpg_autodiff=true)

@test nom_cost(p0) == log_cost(log.(p0))

tr_cost, newp0 = transform_cost(nom_cost, p0, tr)
@test tr_cost(newp0) == log_cost(log.(p0))

grad_holder = deepcopy(p0)
for el in (log_cost, tr_cost)
    el(newp0, grad_holder)
    @test norm(grad_holder) < 1e-3 #0 gradient at minimum
end

H0 = ForwardDiff.hessian(tr_cost, newp0)
mom = 1. 
span = (-10., 10.);

newdp0 = (eigen(H0)).vectors[:, 1]

eprob = specify_curve(log_cost, newp0, newdp0, mom, span);

cb1 = ParameterBounds([1,3], [-10.,-10.], [10.,10.])
cb2 = VerboseOutput(:low, 0.1:2.:10)
cb = CallbackSet(cb1,cb2);
@time mdc = evolve(eprob, Tsit5; callback=cb);


"""
test MDC works and gives reasonable output
"""
@test log_cost(mdc.sol[1][1:3]) < 1e-3
@test log_cost(mdc.sol[end][1:3]) < 1e-3


"""
test injection loss works OK
"""
l2 = build_injection_loss(prob, Tsit5(), tsteps)
@test l2(p0, grad_holder) == 0
@test norm(grad_holder) < 1e-5
