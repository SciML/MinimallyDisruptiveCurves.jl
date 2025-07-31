using ModelingToolkit, OrdinaryDiffEq, ForwardDiff, DiffEqCallbacks, LinearAlgebra, Test

function make_model(input)
    @parameters t
    @parameters k, c, m
    D = Differential(t)
    @variables pos(t) vel(t)

    eqs = [D(pos) ~ vel,
        D(vel) ~ (-1 / m) * (c * vel + k * pos - input(t))
    ]

    ps = [k, c, m] .=> [2.0, 1.0, 4.0]
    ics = [pos, vel] .=> [1.0, 0.0]
    od = ODESystem(eqs, t, first.(ics), first.(ps),
        defaults = merge(Dict(first.(ics) .=> last.(ics)), Dict(first.(ps) .=> last.(ps))),
        name = :mass_spring
    )
    tspan = (0.0, 100.0)
    # prob = ODEProblem(od, ics, tspan, ps)
    return od, ics, tspan, ps
end

od, ics, tspan, ps = make_model(t -> 0.0)

"""
take a log transform of the od parameter space in two ways: at the level of the ODESystem and the level of the ODEProblem
"""

# to_fix = ["c2c","c2","c2a","c3c", "c1c", "a2"]
# tstrct_fix = fix_params(last.(ps), get_name_ids(ps, to_fix))

p0 = last.(ps)
tr = logabs_transform(p0)
log_od = transform_ODESystem(od, tr)
@test typeof(log_od) == ODESystem

prob1 = ODEProblem{true, SciMLBase.FullSpecialize}(od, [], tspan, [])

log_od2, log_ics2,
log_ps2 = transform_problem(
    prob1, tr; unames = ModelingToolkit.get_states(od), pnames = ModelingToolkit.get_ps(od))

"""
check if the two manners of transforming the ODE system give the same output
"""

@test repr.(ModelingToolkit.get_ps(log_od)) == repr.(ModelingToolkit.get_ps(log_od2))

log_prob1 = ODEProblem{true, SciMLBase.FullSpecialize}(log_od, [], tspan, [])
log_prob2 = ODEProblem(log_od2, [], tspan, [])

sol1 = solve(log_prob1, Tsit5())
sol2 = solve(log_prob2, Tsit5())

@test sol1[end] == sol2[end]

"""
check if  log transforming the cost function on od gives the same result as an untransformed cost function on log_od
"""
tsteps = tspan[1]:1.0:tspan[end]
nom_sol = solve(prob1, Tsit5())

function build_loss(which_sol::ODESolution)
    function retf(p)
        sol = solve(which_sol.prob, Tsit5(), p = p, saveat = which_sol.t,
            u0 = convert.(eltype(p), which_sol.prob.u0))
        return sum(sol.u - which_sol.u) do unow
            sum(x -> x^2, unow)
        end
    end
    return retf
end

function build_loss_gradient(which_sol::ODESolution)
    straight_loss = build_loss(which_sol)
    function retf(p, grad)
        ForwardDiff.gradient!(grad, straight_loss, p)
        return straight_loss(p)
    end
    return retf
end

cost1 = build_loss(nom_sol)
cost1_grad = build_loss_gradient(nom_sol)
nom_cost = DiffCost(cost1, cost1_grad)

cost2 = build_loss(sol1)
cost2_grad = build_loss_gradient(sol1)
log_cost = DiffCost(cost2, cost2_grad)

@test nom_cost(p0) == log_cost(log.(p0))

tr_cost, newp0 = transform_cost(nom_cost, p0, tr)
@test tr_cost(newp0) == log_cost(log.(p0))
grad_holder = deepcopy(p0)
g2 = deepcopy(grad_holder)

"""
test that summing losses works
"""
ll = sum_losses([nom_cost, nom_cost], p0)
@test ll(p0 .+ 1.0, grad_holder) == 2nom_cost(p0 .+ 1, g2)
@test grad_holder == 2g2

"""
test gradients of cost functions are zero at minimum as a proxy for correctness of their gradients
"""

for el in (log_cost, tr_cost)
    el(newp0, grad_holder)
    @test norm(grad_holder) < 1e-2 # 0 gradient at minimum
end

"""
test that mdc curve evolves, and listens to mdc_callbacks
"""
H0 = ForwardDiff.hessian(tr_cost, newp0)
mom = 1.0
span = (-2.0, 1.0);

newdp0 = (eigen(H0)).vectors[:, 1]

eprob = MDCProblem(log_cost, newp0, newdp0, mom, span);

cb = [
    Verbose([CurveDistance(0.1:0.1:2.0), HamiltonianResidual(2.3:4:10)]),
    ParameterBounds([1, 3], [-10.0, -10.0], [10.0, 10.0])
]

@time mdc = evolve(eprob, Tsit5; mdc_callback = cb);

"""
check saving callback saves data from interrupted computation. 
Keyword saved_values holds two objects (for two directions) where states and time points are saved. 
"""

span_long = (-20.0, 19.0);
eprob_long = MDCProblem(log_cost, newp0, newdp0, mom, span_long);
cb = [Verbose([CurveDistance(0.1:0.1:2.0)])]

saved_values = (
    SavedValues(Float64, Vector{Float64}), SavedValues(Float64, Vector{Float64}))
mdc = evolve(eprob_long, Tsit5; mdc_callback = cb, saved_values);
#@show saved_values;

"""
check mdc works with mdc_callback vector of subtype T <: CallbackCallable, strict subtype.
"""

cb = [
    Verbose([CurveDistance(0.1:1:10), HamiltonianResidual(2.3:4:10)])
]

@time mdc = evolve(eprob, Tsit5; mdc_callback = cb);

"""
 test MDC works and gives reasonable output
"""

@test log_cost(mdc.sol[1][1:3]) < 1e-3
@test log_cost(mdc.sol[end][1:3]) < 1e-3

"""
test injection loss works OK
"""
l2 = build_injection_loss(prob1, Tsit5(), tsteps)
@test l2(p0, grad_holder) == 0
@test norm(grad_holder) < 1e-5

"""
test fixing parameters (i.e. a transformation that changes the number of parameters) works ok
"""
to_fix = ["c"]
trf = fix_params(last.(ps), get_name_ids(ps, to_fix))
de = MinimallyDisruptiveCurves.transform_ODESystem(od, trf)
@test length(ModelingToolkit.get_ps(de)) == 2

"""
test jumpstarting works
"""
jprob = jumpstart(eprob, 1e-2, true)
mdcj = evolve(jprob, Tsit5; mdc_callback = cb);
@test log_cost(mdcj.sol[1][1:3]) < 1e-3
