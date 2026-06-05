using LinearAlgebra, OrdinaryDiffEq, MinimallyDisruptiveCurves, Statistics, Plots, ForwardDiff, LaTeXStrings
using ModelingToolkit, DifferentiationInterface
using ModelingToolkit: SymbolicT
import ModelingToolkit.t_nounits
import ModelingToolkit.D_nounits


const DI = DifferentiationInterface

# Which eigenvector (direction) to explore along the manifold
which_dir = 2
tspan = (0.0, 10.0) 

@component function LotkaVolterra(;
        name, α = 1.5, β = 1.0, γ = 1.0, δ = 3.0)
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
    push!(initial_conditions, x => 1.0)
    push!(initial_conditions, y => 1.0)

    guesses = Dict{SymbolicT, SymbolicT}()

    eqs = Equation[]
    push!(eqs, D_nounits(x) ~ α * x - β * x * y)
    push!(eqs, D_nounits(y) ~ -δ * y + γ * x * y)

    return System(eqs, t_nounits, vars, params;
        systems = System[], initial_conditions, guesses, name)
end

@named lv = LotkaVolterra()
sys = complete(lv)

# ====================================================================
# --- 2. Build the MTK Mappings and Objective ---
# ====================================================================

println("--- Extracting MTK Structural Layouts ---")

# We build a baseline mapping to easily extract nominal features
mapping = mtk_parameter_mapping(sys; tspan=tspan)
p_nominal = mapping.θ_nominal

# Precompute target baseline features using the mapping's simulation closure
function extract_features_from_sol(sol)
    grid = range(0.0, 10.0, length=200)
    # Lazy interpolation evaluation
    mean_prey = mean(sol(t)[1] for t in grid)
    max_predator = maximum(sol(t)[2] for t in grid)
    return [mean_prey, max_predator]
end

nom_sol = mapping.simulator(p_nominal)
nom_features = extract_features_from_sol(nom_sol)

# Define the unified user cost function that acts on ODESolution
function mtk_user_cost(sol)
    p_features = extract_features_from_sol(sol)
    return sum(abs2, p_features .- nom_features)
end

# Build the unified, backend-prepared cost mapping
cf_result = mtk_cost_mapping(
    sys, 
    mtk_user_cost, 
    AutoForwardDiff(); 
    tspan=tspan
)

core_cost = cf_result.cost_function

# ====================================================================
# --- 3. Execution Pipeline ---
# ====================================================================

println("--- Setting up MTK Lotka-Volterra MDC System ---")

# Compute local Hessian via ForwardDiff on our physical cost function
hess0 = ForwardDiff.hessian(core_cost.f, p_nominal)
eigen_decomposition = eigen(hess0)
init_dir = eigen_decomposition.vectors[:, which_dir]

# Wrap it into an MDCSystem
mdc_sys = MDCSystem(
    core_cost, 
    p_nominal, 
    init_dir, 
    1.0;                  #  Hamiltonian (H)
    names=cf_result.names # Automatically provides [:α, :β, :γ, :δ]
)


stabiliser = mdc_momentum_readjustment(mdc_sys; tol=1e-3)
my_pipeline = CallbackSet(stabiliser)

println("Launching parallel manifold integration...")
mdc_curves = MDCsolve(mdc_sys, span=MDCSpan(-10.0, 10.0); callback = my_pipeline)


# ====================================================================
# --- 4. Animation Pipeline Integration ---
# ====================================================================

println("\nPreparing continuous manifold animation...")

function lotka_volterra_sandbox_painter(θ_physical) # <-- Removed canvas_idx to match the engine
    # 1. Generate uniform time grid for plotting consistency
    plot_t_grid = range(0.0, 10.0, length=200)
    
    # 2. Run your simulators
    sol_nominal = mapping.simulator(p_nominal)
    sol_perturbed = mapping.simulator(θ_physical)
    
    # 3. Sample states strictly on the uniform time grid
    # This prevents the x-axis from dynamically expanding or shifting steps!
    states_nom = [sol_nominal(t) for t in plot_t_grid]
    states_pert = [sol_perturbed(t) for t in plot_t_grid]
    
    prey_nominal = [u[1] for u in states_nom]
    pred_nominal = [u[2] for u in states_nom]
    
    prey_perturbed = [u[1] for u in states_pert]
    pred_perturbed = [u[2] for u in states_pert]

    # Calculate Metrics
    mean_prey_nom  = mean(prey_nominal)
    mean_prey_pert = mean(prey_perturbed)
    
    max_pred_nom   = maximum(pred_nominal)
    max_pred_pert  = maximum(pred_perturbed)

    # 4. Paint to Subplot 1 (The designated simulation sandbox)
    # Background baseline context
    Plots.plot!(
        plot_t_grid, [prey_nominal pred_nominal], 
        subplot = 1, 
        linealpha = 0.20, linestyle = :dash, 
        color = [:blue :red], label = false
    )
    
    # Overlay live trajectory tracking configuration
    Plots.plot!(
        plot_t_grid, [prey_perturbed pred_perturbed], 
        subplot = 1, 
        linewidth = 2, 
        color = [:blue :red], label = ["Prey (x)" "Predator (y)"],
        legend = :topright
    )
    
    # Static Reference Target Lines
    Plots.hline!(
        [mean_prey_nom], 
        subplot = 1, 
        linestyle = :dot, linealpha = 0.4, color = :blue, label = false
    )
    Plots.hline!(
        [max_pred_nom], 
        subplot = 1, 
        linestyle = :dot, linealpha = 0.4, color = :red, label = false
    )
    
    # Dynamic live metric markers
    Plots.hline!(
        [mean_prey_pert], 
        subplot = 1, 
        linestyle = :dashdot, linewidth = 1.2, color = :darkblue, 
        label = "Mean Prey"
    )
    Plots.hline!(
        [max_pred_pert], 
        subplot = 1, 
        linestyle = :dashdot, linewidth = 1.2, color = :darkred, 
        label = "Max Predator"
    )
    
    Plots.plot!(
        subplot = 1,
        xlabel = "Time", ylabel = "Population",
        xlims = (0.0, 10.0), # Firm lock on the physical simulation domain
        ylims = (0.0, 6.0)
    )
end




# Generate the animation object
lv_animation = MinimallyDisruptiveCurves.animate_mdc(
    mdc_curves,
    lotka_volterra_sandbox_painter;
    fps = 20,
    density = 150,  
    raw = true      
)

output_path = joinpath(pwd(), "lotka_volterra_mtk_mdc.gif")
println("Rendering frames and saving video to: $output_path")
Plots.gif(lv_animation, output_path)
