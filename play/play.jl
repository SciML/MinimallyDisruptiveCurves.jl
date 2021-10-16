using OrdinaryDiffEq, ForwardDiff, Statistics, LinearAlgebra

# which md curve to plot
which_dir = 2

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
    sol = solve(prob, Tsit5(); saveat=t)
    return [mean(sol[1,:]), maximum(sol[2,:])]
end

nom_features = features(p)

## loss function, we can take as l2 difference of features vs nominal features
function loss(p)
    prob = remake(nom_prob; p=p)
    p_features = features(p)
    loss = sum(abs2, p_features - nom_features)
    return loss
end

## gradient of loss function
function lossgrad(p, g)
    g[:] = ForwardDiff.gradient(p) do p
        loss(p)
    end
    return loss(p)
end

## package the loss and gradient into a DiffCost structure
cost = DiffCost(loss, lossgrad)

"""
We evaluate the hessian once only, at p.
Why? to find locally insensitive directions of parameter perturbation
The small eigenvalues of the Hessian are one easy way of defining these directions 
"""
hess0 = ForwardDiff.hessian(loss, p)
ev(i) = -eigen(hess0).vectors[:,i]

init_dir = ev(which_dir); momentum = 1.; span = (-15., 15.)
curve_prob = MDCProblem(cost, p, init_dir, momentum, span)

map(1:10) do i
    curve_prob_orig = curveProblem(cost, p, init_dir, momentum, span)
    @time mdc2 = evolve(curve_prob_orig, Tsit5);
    @time mdc = evolve(curve_prob, Tsit5);
    return cost_trajectory(mdc, mdc.sol.t) |> mean, cost_trajectory(mdc2, mdc2.sol.t) |> mean
end


function sol_at_p(p)
    prob = remake(nom_prob; p=p)
    sol = solve(prob, Tsit5())
end

# p1 = plot(mdc; pnames=[L"p_1" L"p_2" L"p_3" L"p_4"])

# cost_vec = [mdc.cost(el) for el in eachcol(trajectory(mdc))]
# p2 = plot(distances(mdc), log.(cost_vec), ylabel="log(cost)", xlabel="distance", title="cost over MD curve");

# mdc_plot = plot(p1, p2, layout=(2, 1), size=(800, 800))

# nominal_trajectory = plot(sol_at_p(mdc(0.)[:states]), label=["prey" "predator"])
# perturbed_trajectory = plot(sol_at_p(mdc(-15.)[:states]), label=["prey" "predator"])
# traj_comparison = plot(nominal_trajectory, perturbed_trajectory, layout=(2, 1), xlabel="time", ylabel="population")

# Lessons