using ModelingToolkit
using ModelingToolkit: SymbolicT
import ModelingToolkit.t_nounits
import ModelingToolkit.D_nounits
using MinimallyDisruptiveCurves
using OrdinaryDiffEq
using ADTypes 
using ForwardDiff


@component function LotkaVolterra(;
        name, α = 1.3, β = 0.9, γ = 0.8, δ = 1.8)
    @parameters begin
        α = α
        β = β
        γ = γ
        δ = δ
    end
    params = SymbolicT[]
    push!(params, α)
    push!(params, β)
    push!(params, γ)
    push!(params, δ)

    @variables begin
        x(t_nounits)
        y(t_nounits)
    end
    vars = SymbolicT[]
    push!(vars, x)
    push!(vars, y)

    initial_conditions = Dict{SymbolicT, SymbolicT}()
    push!(initial_conditions, x => 3.1)
    push!(initial_conditions, y => 1.5)

    guesses = Dict{SymbolicT, SymbolicT}()

    eqs = Equation[]
    push!(eqs, D_nounits(x) ~ α * x - β * x * y)
    push!(eqs, D_nounits(y) ~ -δ * y + γ * x * y)

    return System(eqs, t_nounits, vars, params;
        systems = System[], initial_conditions, guesses, name)
end

@named lv = LotkaVolterra()
sys = complete(lv)

# 2. Define the user cost function 
# Example: Tracks a target prey population 'x' close to 2.0 over the trajectory
function my_user_cost(sol)
    return sum(abs2, [state[1] - 2.0 for state in sol.u]) / length(sol.u)
end

cf_result = mtk_cost_mapping(sys, my_user_cost, AutoForwardDiff(); tspan=(0.0, 10.0))

physical_cost  = cf_result.cost_function
physical_theta = cf_result.θ_nominal
physical_names = cf_result.names

println("--- Physical Space Pipeline ---")
println("Physical Names: ", physical_names)
println("Physical Baseline Vector (θ): ", physical_theta)

# 5. Construct a Transformation Chain from your package
# We freeze γ (index 3) at its baseline, and sweep α, β, and δ in log-space.
free_indices = [1, 2, 4]
fixed_value  = [physical_theta[3]] 

mask_t = FixedParamsTransform(free_indices, fixed_value, length(physical_theta))
log_t  = LogAbsTransform()
chain  = TransformChain(mask_t, log_t)

# 6. Wrap physical cost into your native TransformedCost structure
optimizer_cost = TransformedCost(physical_cost, chain)
optimizer_names = MinimallyDisruptiveCurves.transform_names(chain, physical_names)

println("\n--- Parameter Space Transformations ---")
println("Optimizer Parameter Names: ", optimizer_names)

# Map nominal parameters from physical space -> optimizer space coordinates
θ_optimizer = MinimallyDisruptiveCurves.inverse(chain, physical_theta)
println("Optimizer Space Coordinates: ", θ_optimizer)

# 7. Preallocate non-allocating loop execution buffers 
gz = similar(physical_theta)
gθ = similar(θ_optimizer)

# 8. Run the final evaluation test (Simultaneous value and gradient pullback)
cost_value = optimizer_cost(θ_optimizer, gθ, gz)

println("\n--- Output Verification ---")
println("Calculated Cost: ", cost_value)
println("Gradients backprop'd to optimizer space (gθ): ", gθ)
