using LinearAlgebra, OrdinaryDiffEq, MinimallyDisruptiveCurves, ForwardDiff, Plots

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

"""
    simulation(θ; u0=[1.0, 0.0], tspan=(0.0, 10.0), saveat=0.1)

Standalone simulation call. Promotes initial conditions `u0` automatically 
to match the type of `θ` (essential for ForwardDiff compatibility).
"""
function simulation(θ; u0=[1.0, 0.0], tspan=(0.0, 10.0), saveat=0.1)
    # Automatically promote state vector types to handle ForwardDiff Dual numbers cleanly
    T = eltype(θ)
    u0_typed = convert(Vector{T}, u0)
    
    prob = ODEProblem(mass_spring_dynamics!, u0_typed, tspan, θ)
    return solve(prob, Tsit5(), saveat=saveat)
end

# ====================================================================
# --- Dynamic Cost Function Factory ---
# ====================================================================

"""
    make_mse_cost_function(θ_nominal; u0=[1.0, 0.0], tspan=(0.0, 10.0), dt=0.1)

Generates a clean `CostFunction` instance using the standalone simulation utility.
"""
function make_mse_cost_function(θ_nominal; u0=[1.0, 0.0], tspan=(0.0, 10.0), dt=0.1)
    # 1. Generate the immutable reference trajectory data using our standalone simulator
    sol_nominal = simulation(θ_nominal; u0=u0, tspan=tspan, saveat=dt)
    
    target_times = sol_nominal.t
    target_positions = [sol[1] for sol in sol_nominal.u]
    
    # 2. Define the objective function closure (f)
    function f(θ)
        # Prevent unphysical negative parameters or division by zero
        if any(θ .<= 1e-3)
            return 100.0 + sum(abs2, min.(0.0, θ))
        end
        
        # Call standalone simulation, evaluating at exact target timesteps
        sol = simulation(θ; u0=u0, tspan=tspan, saveat=target_times)
       
        current_positions = [s[1] for s in sol.u]
        return sum(abs2, current_positions .- target_positions) / length(target_times)
    end
    
    # 3. Define the gradient closure (grad!)
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

θ_nominal = [1.0, 2.5, 5.0]     
dθ_nominal = [1.0, 2.0, 3.0]     
u0_physical = [1.0, 0.0]        
tspan_physical = (0.0, 10.0)    

# 1. Build the Cost Function
core_cost = make_mse_cost_function(θ_nominal, u0=u0_physical, tspan=tspan_physical)

# 2. Wire up the Transform Chain
chain = TransformChain() 
transformed_cost = TransformedCost(core_cost, chain)

# 3. Instantiate the MDC System
θ₀ = θ_nominal
dθ₀ = θ_nominal
H = 2.0              

sys = MDCSystem(transformed_cost, θ₀, dθ₀, H)

# 4. Bind stabilization callbacks
stabiliser = mdc_momentum_readjustment(sys; tol=1e-3)
my_pipeline = CallbackSet(stabiliser)

# 5. Run curve tracing
@time mdc_curves = MDCsolve(sys, span=MDCSpan(-5.0, 15.0), callback=my_pipeline)

# ====================================================================
# --- Verification & Analysis ---
# ====================================================================

println("\n--- Exploration Metrics ---")
if mdc_curves.positive_sol !== nothing
    final_state = mdc_curves.positive_sol.u[end]
    θ_explored = final_state[1:3]
    
    println("Initial Params (θ₀):          ", round.(θ₀, digits=4))
    println("Explored Params (MDC End):   ", round.(θ_explored, digits=4))
    
    println("\nValidating Ratio Preservation:")
    println("Initial c/m: ", round(θ₀[2]/θ₀[1], digits=4), "  |  Explored c/m: ", round(θ_explored[2]/θ_explored[1], digits=4))
    println("Initial k/m: ", round(θ₀[3]/θ₀[1], digits=4), "  |  Explored k/m: ", round(θ_explored[3]/θ_explored[1], digits=4))
    
    final_cost = core_cost.f(θ_explored)
    println("\nMSE Cost Relative to Nominal Trajectory: ", round(final_cost, digits=7))
    
else
    println("Error: MDC system integration yielded no solutions.")
end
