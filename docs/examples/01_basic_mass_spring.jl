ENV["GKSwstype"] = "100"; 
# # Basic Mass-Spring System
#
# This example shows how Minimally Disruptive Curves (MDC) finds a structural 
# unidentifiability in the basic damped harmonic oscillator model (where `m` is mass, 
# `c` is damping, and `k` is the spring constant). We will see that the trajectory 
# doesn't vary where the ratios `c/m` and `k/m` remain constant. This is obvious algebraically, as the equations are $\dot{v} = -(c/m)v - (k/m)x$.
#
# !!! note
#     This script does not explore how to initialise a good starting MDC direction. 
#     We make a lucky guess here. That's the topic of the next Lotka-Volterra example (and the NFKB example)

using LinearAlgebra, OrdinaryDiffEq, MinimallyDisruptiveCurves, Plots
using ForwardDiff

# ## Simulator
# We define the standard 2D mass-spring-damper vector field: $\dot{x} = v$, $\dot{v} = -(c/m)v - (k/m)x$.

function mass_spring_dynamics!(du, u, p, t)
    m, c, k = p
    position, velocity = u[1], u[2]

    du[1] = velocity
    du[2] = -(c / m) * velocity - (k / m) * position
    return nothing
end

# ## Dynamic Cost Function Factory
# This factory generates a `CostFunction` instance. It pre-computes a reference 
# trajectory using `θ_nominal` and evaluates the Mean Squared Error (MSE) deviation for 
# any test parameter vector `θ`.

function make_mse_cost_function(θ_nominal; u0 = [1.0, 0.0], tspan = (0.0, 10.0), dt = 0.1)
    #1. Generate the immutable reference trajectory data
    prob_nominal = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_nominal)
    sol_nominal = solve(prob_nominal, Tsit5(), saveat = dt)

    target_times = sol_nominal.t
    target_positions = [sol[1] for sol in sol_nominal.u]

    #2. Define the objective function closure (f)
    function f(θ)
        if any(θ .<= 1.0e-3)
            #Ensure the penalty return type matches the input dual/real element type
            return 100.0 + sum(abs2, min.(zero(eltype(θ)), θ))
        end

        prob = ODEProblem(mass_spring_dynamics!, u0, tspan, θ)
        sol = solve(prob, Tsit5(), saveat = target_times)

        current_positions = [s[1] for s in sol.u]
        return sum(abs2, current_positions .- target_positions) / length(target_times)
    end

    #3. Define the exact Automatic Differentiation gradient closure (grad!)
    function grad!(g, θ)
        ForwardDiff.gradient!(g, f, θ)
        return g
    end

    return CostFunction(f, grad!)
end

# ## Execution Pipeline
# Let's define our base system nominal profile and instantiate the MDC System.

θ_nominal = [1.0, 0.5, 5.0]     #m = 1.0, c = 0.5, k = 5.0
dθ_nominal = θ_nominal          #Lucky guess for initial direction
u0_physical = [1.0, 0.0]        #Initial position=1, velocity=0
tspan_physical = (0.0, 10.0)    #Observe for 10 seconds

# Build the Cost Function mapping parameter variants against the baseline
core_cost = make_mse_cost_function(θ_nominal, u0 = u0_physical, tspan = tspan_physical)

# Wire up the Transform Chain (empty in this case, and could directly pass core_cost to the `MDCProblem`)
chain = TransformChain()
transformed_cost = TransformedCost(core_cost, chain)

# Instantiate the MDC System
θ₀ = θ_nominal
dθ₀ = θ_nominal
H = 1.0                        #Hamiltonian: think of as momentum of the trajectory

sys = MDCProblem(transformed_cost, θ₀, dθ₀, H; names = [:mass, :damping, :stiffness])

# Setup the standard stability callback and solve
stabilizer = mdc_momentum_readjustment(sys; tol = 1.0e-3)
my_pipeline = CallbackSet(stabilizer)

mdc_curves = MDCSolve(sys, span = MDCSpan(-5.0, 5.0), callback = my_pipeline; mode = :fast)

# ## Verification & Analysis
# Let's inspect the parameters discovered at the endpoint of the curve.

final_state = mdc_curves.negative_sol.u[end]
θ_explored = final_state[1:3]

println("Initial Params (θ₀):          ", round.(θ₀, digits = 4))
println("Explored Params (MDC End):   ", round.(θ_explored, digits = 4))

println("\nValidating Ratio Preservation:")
println("Initial c/m: ", round(θ₀[2] / θ₀[1], digits = 4), "  |  Explored c/m: ", round(θ_explored[2] / θ_explored[1], digits = 4))
println("Initial k/m: ", round(θ₀[3] / θ₀[1], digits = 4), "  |  Explored k/m: ", round(θ_explored[3] / θ_explored[1], digits = 4))

# Calculate the exact MSE cost at the terminal point of our parameter path
final_cost = core_cost.f(θ_explored)
println("\nMSE Cost Relative to Nominal Trajectory: ", round(final_cost, digits = 7))

# ## Animation Custom Visualization
# We can define a custom visualization function to pass to `animate_mdc`, 
# which paints directly onto the plot canvas at each frame.

function animate_system_response(θ_current)
    u0 = [1.0, 0.0]
    tspan = (0.0, 10.0)
    dt = 0.1

    prob_nominal = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_nominal)
    sol_nominal = solve(prob_nominal, Tsit5(), saveat = dt)

    prob_current = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_current)
    sol_current = solve(prob_current, Tsit5(), saveat = dt)

    pos_nominal = [u[1] for u in sol_nominal.u]
    pos_current = [u[1] for u in sol_current.u]

    #Plot the ground-truth nominal response as a static dashed black line
    Plots.plot!(
        sol_nominal.t, pos_nominal,
        subplot = 1,
        line = (:black, :dash),
        linewidth = 2,
        label = "Nominal Target"
    )

    #Overlay the explored parameter trajectory response in solid blue
    return Plots.plot!(
        sol_current.t, pos_current,
        subplot = 1,
        color = :blue,
        linewidth = 2.5,
        label = "MDC Configuration Response",
        xlabel = "Physical Time (s)",
        ylabel = "Position (x)",
        ylim = (-1.2, 1.2)
    )
end

# ## Animation
# We can generate an animation using `animate_mdc` and save it as a gif:

anim = animate_mdc(mdc_curves, animate_system_response; density = 100, fps = 15, raw = true)
gif(anim, "mass_spring_mdc_sweep.gif", fps = 15);

#md # ![Mass Spring MDC Sweep](mass_spring_mdc_sweep.gif)

