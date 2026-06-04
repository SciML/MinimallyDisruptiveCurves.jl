using LinearAlgebra, OrdinaryDiffEq, MinimallyDisruptiveCurves, Plots
using ForwardDiff


# ====================================================================
# --- Core Physics Engine ---
# ====================================================================

# Standard 2D mass-spring-damper vector field: ẋ = v, v̇ = -(c/m)v - (k/m)x
function mass_spring_dynamics!(du, u, p, t)
    m, c, k = p
    position, velocity = u[1], u[2]
    
    du[1] = velocity
    du[2] = -(c / m) * velocity - (k / m) * position
    return nothing
end

# ====================================================================
# --- Dynamic Cost Function Factory ---
# ====================================================================

"""
    make_mse_cost_function(θ_nominal; u0=[1.0, 0.0], tspan=(0.0, 10.0), dt=0.1)

Generates a clean `CostFunction` instance. It pre-computes a reference trajectory 
using `θ_nominal` and evaluates the MSE deviation for any test parameter vector θ.
"""
function make_mse_cost_function(θ_nominal; u0=[1.0, 0.0], tspan=(0.0, 10.0), dt=0.1)
    # 1. Generate the immutable reference trajectory data
    prob_nominal = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_nominal)
    sol_nominal = solve(prob_nominal, Tsit5(), saveat=dt)
    
    target_times = sol_nominal.t
    target_positions = [sol[1] for sol in sol_nominal.u]
    
    # 2. Define the objective function closure (f)
    # NOTE: We remove the manual Float64 type-restrictions so Dual numbers can pass through
    function f(θ)
        if any(θ .<= 1e-3)
            # Ensure the penalty return type matches the input dual/real element type dynamically
            return 100.0 + sum(abs2, min.(zero(eltype(θ)), θ))
        end
        
        # Pass θ directly—OrdinaryDiffEq automatically handles dual-number parameters!
        prob = ODEProblem(mass_spring_dynamics!, u0, tspan, θ)
        sol = solve(prob, Tsit5(), saveat=target_times)
       
        current_positions = [s[1] for s in sol.u]
        return sum(abs2, current_positions .- target_positions) / length(target_times)
    end
    
    # 3. Define the exact Automatic Differentiation gradient closure (grad!)
    function grad!(g, θ)
        ForwardDiff.gradient!(g, f, θ)
        return g
    end
    
    return CostFunction(f, grad!)
end

# ====================================================================
# --- Execution Pipeline ---
# ====================================================================

println("--- Setting up Mass-Spring MDC Test ---")

# Define our base physical system nominal profile
θ_nominal = [1.0, 0.5, 5.0]     # True baseline: m=2.0, c=1.5, k=8.0
dθ_nominal = [1.0, 2.0, 3.0]     # True baseline: m=2.0, c=1.5, k=8.0
u0_physical = [1.0, 0.0]        # Initial position=1, velocity=0
tspan_physical = (0.0, 10.0)    # Observe for 10 seconds

# 1. Build the Cost Function mapping parameter variants against the baseline
core_cost = make_mse_cost_function(θ_nominal, u0=u0_physical, tspan=tspan_physical)

# 2. Wire up the Transform Chain
chain = TransformChain() 
transformed_cost = TransformedCost(core_cost, chain)

# 3. Instantiate the MDC System
# Let's start the exploration at a scaled configuration where c/m and k/m match perfectly.
# This means the initial cost should be mathematically zero!
θ₀ = θ_nominal
dθ₀ = θ_nominal

H = 1.0              # Parameter exploration kinetic energy threshold

sys = MDCSystem(transformed_cost, θ₀, dθ₀,H; names=[:mass, :damping, :stiffness])


# safety_cb  = mdc_safety_callback(sys)
stabilizer = mdc_momentum_readjustment(sys; tol=1e-3)
my_pipeline = CallbackSet(stabilizer)



mdc_curves = MDCsolve(sys, span=MDCSpan(-5.0, 5.0), callback=my_pipeline)


# ====================================================================
# --- Verification & Analysis ---
# ====================================================================

println("\n--- Exploration Metrics ---")
if mdc_curves.positive_sol !== nothing
    final_state = mdc_curves.negative_sol.u[end]
    θ_explored = final_state[1:3]
    
    println("Initial Params (θ₀):          ", round.(θ₀, digits=4))
    println("Explored Params (MDC End):   ", round.(θ_explored, digits=4))
    
    println("\nValidating Ratio Preservation:")
    println("Initial c/m: ", round(θ₀[2]/θ₀[1], digits=4), "  |  Explored c/m: ", round(θ_explored[2]/θ_explored[1], digits=4))
    println("Initial k/m: ", round(θ₀[3]/θ₀[1], digits=4), "  |  Explored k/m: ", round(θ_explored[3]/θ_explored[1], digits=4))
    
    # Calculate the exact MSE cost at the terminal point of our parameter path
    final_cost = core_cost.f(θ_explored)
    println("\nMSE Cost Relative to Nominal Trajectory: ", round(final_cost, digits=7))
else
    println("Error: MDC system integration yielded no solutions.")
end


# ====================================================================
# --- Animation Custom Visualization ---
# ====================================================================

"""
    animate_system_response(plt, θ_current)

A user custom visualization function passed to `animate_mdc`.
- `plt`: The ongoing animation plot canvas handle.
- `θ_current`: The physical parameters [m, c, k] supplied at the current path step.
"""
function animate_system_response(plt, θ_current)
    # 1. Setup simulation configuration matching your setup
    u0 = [1.0, 0.0]
    tspan = (0.0, 10.0)
    dt = 0.1
    
    # 2. Simulate the nominal baseline trajectory for comparison
    prob_nominal = ODEProblem(mass_spring_dynamics!, u0, tspan, [1.0, 2.5, 5.0]) # θ_nominal
    sol_nominal = solve(prob_nominal, Tsit5(), saveat=dt)
    
    # 3. Simulate the system under the current explored parameter set
    prob_current = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_current)
    sol_current = solve(prob_current, Tsit5(), saveat=dt)
    
    # 4. Extract position coordinates (index 1)
    pos_nominal = [u[1] for u in sol_nominal.u]
    pos_current = [u[1] for u in sol_current.u]
    
    # 5. Paint directly onto Subplot 2
    # Plot the ground-truth nominal response as a static dashed black line
    Plots.plot!(
        plt, sol_nominal.t, pos_nominal,
        subplot = 2,
        line = (:black, :dash),
        linewidth = 2,
        label = "Nominal Target"
    )
    
    # Overlay the explored parameter trajectory response in solid blue
    Plots.plot!(
        plt, sol_current.t, pos_current,
        subplot = 2,
        color = :blue,
        linewidth = 2.5,
        label = "MDC Configuration Response",
        xlabel = "Physical Time (s)",
        ylabel = "Position (x)",
        ylim = (-1.2, 1.2) # Bound axes to stop shifting during playback
    )
end

# --- Trigger the Animation Generation ---
# println("\n--- Generating MDC Parameter Sweep Animation ---")
# anim = animate_mdc(mdc_curves, animate_system_response; density=100, fps=15, raw=true)

# Save out your animation file
# gif(anim, "mass_spring_mdc_sweep.gif", fps=15)
