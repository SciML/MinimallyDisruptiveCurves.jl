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
    od = ODESystem(eqs, t, first.(ics), first.(ps), defaults=merge(Dict(first.(ics) .=> last.(ics)), Dict(first.(ps) .=> last.(ps))), name=:mass_spring
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

prob1 = ODEProblem(od, [], tspan, [])

"""
check if the two manners of transforming the ODE system give the same output
"""

log_prob1 = ODEProblem{true,SciMLBase.FullSpecialize}(log_od, [], tspan, [])
log_sol1 = solve(log_prob1, Tsit5())


"""
check if  log transforming the cost function on od gives the same result as an untransformed cost function on log_od
"""
tsteps = tspan[1]:1.0:tspan[end]


nom_sol = solve(prob1, Tsit5())

grad_holder = deepcopy(p0)
g2 = deepcopy(grad_holder)


function build_loss(p, which_sol::ODESolution)
    sol = solve(which_sol.prob, Tsit5(), p=p, saveat=which_sol.t, u0=convert.(eltype(p), which_sol.prob.u0))
    return sum(sol.u - which_sol.u) do unow
        sum(x -> x^2, unow)
    end
end


nom_cost = p -> build_loss(p, log_sol1)
tr_cost, newp0 = transform_cost(nom_cost, p0, tr)

function build_loss_gradient(p, grad, which_sol::ODESolution)
    straight_loss(p) = build_loss(p, which_sol)
    ForwardDiff.gradient!(grad, straight_loss,p)
    return straight_loss(p)
end

nom_cost_grad(p, grad) = build_loss_gradient(p, grad, log_sol1)

my_loss = DiffCost(nom_cost, nom_cost_grad)

H0 = ForwardDiff.hessian(my_loss, newp0)
mom = 20.0
span = (-10.0, 10.0);

newdp0 = (eigen(H0)).vectors[:, 1]

eprob = MDCProblem(my_loss, newp0, newdp0, mom, span);

cb = [
    Verbose([CurveDistance(0.1:0.1:2.0), HamiltonianResidual(0.1:0.1:2.)]),
    ParameterBounds([1, 3], [-10.0, -10.0], [10.0, 10.0])
]


@time mdc = evolve(eprob, Tsit5; mdc_callback=cb);