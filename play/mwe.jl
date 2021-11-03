using OrdinaryDiffEq, ForwardDiff, Statistics, LinearAlgebra

# which md curve to plot
which_dir = 1

## define dynamics of differential equation
function f(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2] # prey
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2] # predator
end

u0 = [1.0;1.0] # initial populations
tspan = (0.0, 10.0) # time span to simulate over
t = collect(range(0, stop=10., length=200)) # time points to measure
p = [1.5,1.0,3.0,1.0] # initial parameter values
nom_prob = ODEProblem(f, u0, tspan, p) # package as an ODE problem
nom_sol = solve(nom_prob, Tsit5()) # solve 


## Model features of interest are mean prey population, and max predator population (over time)
function features(p)
    prob = remake(nom_prob; p=p)
    sol = solve(prob, Vern9(); saveat=t)
    return [mean(sol[1,:]), maximum(sol[2,:])]
end

nom_features = features(p)

function loss(p)
    prob = remake(nom_prob; p=p)
    p_features = features(p)
    loss = sum(abs2, p_features - nom_features)
    return loss
end
function lossgrad(p, g)
    g[:] = ForwardDiff.gradient(p) do p
        loss(p)
    end
    return loss(p)
end
cost = DiffCost(loss, lossgrad)
hess0 = ForwardDiff.hessian(loss, p)
ev(i) = -eigen(hess0).vectors[:,i]

init_dir = ev(which_dir); momentum = 1. ; span = (-15., 15.)
curve_prob = MDCProblem(cost, p, init_dir, momentum, span)

@time mdc = evolve(curve_prob, Tsit5; mdc_callback=cb);
