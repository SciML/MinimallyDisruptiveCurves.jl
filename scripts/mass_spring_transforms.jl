using LinearAlgebra, OrdinaryDiffEq, MinimallyDisruptiveCurves, Plots
using ForwardDiff

# ====================================================================
# --- Core Physics Engine ---
# ====================================================================

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

println("--- Setting up Mass-Spring Chained MDC Test ---")

# Define baseline physical profiles
θ_nominal = [1.0, 0.5, 5.0]     # m = 1.0, c = 0.5, k = 5.0
u0_physical = [1.0, 0.0]
tspan_physical = (0.0, 10.0)

# 1. Build core underlying physical model cost
core_cost = make_mse_cost_function(θ_nominal, u0=u0_physical, tspan=tspan_physical)

# 2. Wire up the Chained Mathematical Transformations
# To explore in LOG space on the FREE indices, wrap the LogAbs transform *inside* the active coordinates.
# This means our solver runs on [log(c), log(k)].
# forward direction: [log(c), log(k)] -> [c, k] -> [1.0, c, k]
fix_transform = FixedParamsTransform([2, 3], [1.0], 3)
chain = TransformChain(LogAbsTransform(), fix_transform)
 
transformed_cost = TransformedCost(core_cost, chain)

# 3. Instantiate the operational parameter states
θ₀ = MinimallyDisruptiveCurves.inverse(chain, θ_nominal)  # Resolves exactly to 2 elements: [log(0.5), log(5.0)]
dθ₀ = [1.0, 1.0]                # 2-element directional push vector matching internal dimension

H = 1.0 # Exploration energy headroom limit

# Pass baseline physical metadata names into the constructor tracking framework
sys = MDCSystem(
    transformed_cost, 
    θ₀, 
    dθ₀, 
    H; 
    names=[:mass, :damping, :stiffness]
)

# 4. Invoke Multi-Threaded Exploration Engine
stabilizer = mdc_momentum_readjustment(sys; tol=1e-3)
my_pipeline = CallbackSet(stabilizer)

println("Launching parallel manifold integration...")
mdc_curves = MDCSolve(sys, span=MDCSpan(-5.0, 5.0), callback=my_pipeline)

# ====================================================================
# --- Verification & Analysis ---
# ====================================================================

println("\n--- Operational Space Exploration Metrics ---")
println("Evaluated Active Parameter Labels: ", sys.names)

if mdc_curves.positive_sol !== nothing && mdc_curves.negative_sol !== nothing
    # Extract the final terminal state coordinates found along the negative integration trace
    final_operational_state = mdc_curves.negative_sol.u[end]
    θ_operational_end = final_operational_state[1:length(sys.θ₀)]
    
    # Run parameters backward through inverse maps to map coordinates back to physical units
    # FIXED: Restored to forward map layout to inflate active solver values back to 3D physical domain
    θ_physical_end = forward(chain, θ_operational_end)
    
    println("\nInitial Physical Configuration: ", round.(θ_nominal, digits=4))
    println("Terminal Physical Configuration: ", round.(θ_physical_end, digits=4))
    
    println("\n--- Subspace Constraint Verification ---")
    println("Did Mass stay completely fixed? ", θ_physical_end[1] == θ_nominal[1] ? "YES (1.0)" : "NO")
    
    final_cost = core_cost.f(θ_physical_end)
    println("MSE Loss at Path Endpoint: ", round(final_cost, digits=7))
else
    println("Error: MDC integration failed.")
end
