# ====================================================================
# MINIMALLY DISRUPTIVE CURVES (MDC) — LOTKA-VOLTERRA MANIFOLD TRACE
# ====================================================================
# This script constructs a component-based Lotka-Volterra ecosystem model
# using ModelingToolkit (MTK), sets up an objective cost function measuring
# macro-level system features, and solves for unidentifiable parameter
# combinations using Minimally Disruptive Curves.

using LinearAlgebra
using OrdinaryDiffEq
using MinimallyDisruptiveCurves
using Statistics
using Plots
using ForwardDiff
using ModelingToolkit
import ModelingToolkit: SymbolicT
using DifferentiationInterface
import SciMLBase: successful_retcode
import ModelingToolkit.t_nounits
import ModelingToolkit.D_nounits

# Global Configuration Parameters
const TSPAN_PHYSICAL = (0.0, 10.0)  # Time horizon for the differential equations
const EXPLORATION_DIR = 2           # Hessian eigenvector index selected for manifold path initialization

# ====================================================================
# 1. MODEL DEFINITION via ModelingToolkit (MTK)
# ====================================================================

"""
    LotkaVolterra(; name, α=1.5, β=1.0, γ=1.0, δ=3.0)

Constructs a standard component-based Predator-Prey model system container.
States: 
  - `x(t)`: Prey population 
  - `y(t)`: Predator population
Parameters:
  - `α`: Prey birth rate          - `β`: Predation consumption rate
  - `γ`: Predator efficiency growth - `δ`: Predator natural death rate
"""
@component function LotkaVolterra(;
        name, α = 1.3, β = 0.9, γ = 0.8, δ = 1.8
    )
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

    return System(
        eqs, t_nounits, vars, params;
        systems = System[], initial_conditions, guesses, name
    )
end

@named lv = LotkaVolterra()
sys = complete(lv)

# ====================================================================
# 2. HARDENED OBJECTIVE FUNCTION AND NON-BLOCKING STEP SAFEGUARDS
# ====================================================================
println("--- Extracting MTK Structural Layouts ---")

# Map MTK model to a simulation that depends on a flat parameter vector.
mapping = mtk_parameter_mapping(sys; tspan = TSPAN_PHYSICAL)
p_nominal = mapping.θ_nominal

# Precompute target baseline features using nominal trajectory mapping
nom_sol = mapping.simulator(p_nominal)
const grid_steps = range(TSPAN_PHYSICAL[1], TSPAN_PHYSICAL[2], length = 200)
const nom_features = [
    mean(nom_sol(t)[1] for t in grid_steps),
    maximum(nom_sol(t)[2] for t in grid_steps),
]

"""
    mtk_user_cost(sol)

High-performance cost function. Uses pure branching condition checks to catch 
solver stalls (MaxIters), instability, or explosions, instantly returning a 
stable numerical penalty wall (10000.0) without try-catch overhead.
"""
function mtk_user_cost(sol)
    # 1. Intercept solver abort status (MaxIters, DtNaN, Unstable, etc.) instantly
    if !successful_retcode(sol.retcode)
        return 10000.0
    end

    # 2. Sample states along the time grid array
    vals = [sol(t) for t in grid_steps]

    # 3. Short-circuit check for numerical NaN/Inf explosions inline
    if any(any(isnan.(v)) || any(isinf.(v)) for v in vals)
        return 10000.0
    end

    # 4. If simulation passed validation tests, evaluate feature error metrics
    mean_prey = mean(v[1] for v in vals)
    max_predator = maximum(v[2] for v in vals)

    return sum(abs2, [mean_prey, max_predator] .- nom_features)
end

# Compile the user metric objective function into an AD-ready configuration layout
cf_result = mtk_cost_mapping(
    sys,
    mtk_user_cost,
    AutoForwardDiff();
    tspan = TSPAN_PHYSICAL,
)
core_cost = cf_result.cost_function

# ====================================================================
# 3. MDC GENERATION PIPELINE
# ====================================================================
println("--- Setting up MTK Lotka-Volterra MDC System ---")

# Compute the local curvature Hessian matrix to detect directions of initial insensitivity
hess0 = ForwardDiff.hessian(core_cost.f, p_nominal)
eigen_decomposition = eigen(hess0)

# Extract selected eigenvector corresponding to initial curve direction
init_dir = eigen_decomposition.vectors[:, EXPLORATION_DIR]

# Build MDC system
mdc_sys = MDCSystem(
    core_cost,
    p_nominal,
    init_dir,
    1.0;                 #  Hamiltonian / momentum (H)
    names = cf_result.names # Preserves mapping symbols: [:α, :β, :γ, :δ]
)

# Attach conservation stabilization callback routine to mitigate integration drift
stabiliser = mdc_momentum_readjustment(mdc_sys; tol = 1.0e-3)
my_pipeline = CallbackSet(stabiliser)

println("Launching MDC...")
@time mdc_curves = MDCSolve(mdc_sys, span = MDCSpan(-5.0, 5.0); callback = my_pipeline)

# ====================================================================
# 4. CUSTOM VISUALIZATION AND ANIMATION ENGINE INTERACTION
# ====================================================================
println("\nPreparing continuous manifold animation...")

"""
    lotka_volterra_sandbox_painter(θ_physical)

Plots the live system dynamics at parameter configuration `θ_physical`. Plots are 
evaluated on a strict grid to preserve structural time-axis scaling during rendering.
"""
function lotka_volterra_sandbox_painter(θ_physical)
    # 1. Enforce a locked uniform axis domain grid to prevent visual stretching
    plot_t_grid = range(TSPAN_PHYSICAL[1], TSPAN_PHYSICAL[2], length = 200)

    # 2. Evaluate simulated trajectories
    sol_nominal = mapping.simulator(p_nominal)
    sol_perturbed = mapping.simulator(θ_physical)

    # 3. Slice state vectors systematically over the static domain grid
    states_nom = [sol_nominal(t) for t in plot_t_grid]
    states_pert = [sol_perturbed(t) for t in plot_t_grid]

    prey_nominal = [u[1] for u in states_nom]
    pred_nominal = [u[2] for u in states_nom]

    prey_perturbed = [u[1] for u in states_pert]
    pred_perturbed = [u[2] for u in states_pert]

    # Compute descriptive system properties
    mean_prey_nom = mean(prey_nominal)
    mean_prey_pert = mean(prey_perturbed)
    max_pred_nom = maximum(pred_nominal)
    max_pred_pert = maximum(pred_perturbed)

    # 4. Generate visual composition layers inside Subplot 1
    # Plot baseline reference context (Static faint indicators)
    Plots.plot!(
        plot_t_grid, [prey_nominal pred_nominal],
        subplot = 1,
        linealpha = 0.2, linestyle = :dash,
        color = [:blue :red], label = false
    )

    # Overlay the current explored parameters system response
    Plots.plot!(
        plot_t_grid, [prey_perturbed pred_perturbed],
        subplot = 1,
        linewidth = 2,
        color = [:blue :red], label = ["Prey (x)" "Predator (y)"],
        legend = :topright
    )

    # Draw nominal target feature constraints
    Plots.hline!([mean_prey_nom], subplot = 1, linestyle = :dot, linealpha = 0.4, color = :blue, label = false)
    Plots.hline!([max_pred_nom], subplot = 1, linestyle = :dot, linealpha = 0.4, color = :red, label = false)

    # Draw dynamic tracked feature shifts
    Plots.hline!([mean_prey_pert], subplot = 1, linestyle = :dashdot, linewidth = 1.2, color = :darkblue, label = "Mean Prey")
    Plots.hline!([max_pred_pert], subplot = 1, linestyle = :dashdot, linewidth = 1.2, color = :darkred, label = "Max Predator")

    # Format Subplot Canvas Properties
    return Plots.plot!(
        subplot = 1,
        xlabel = "Time", ylabel = "Population",
        xlims = TSPAN_PHYSICAL,
        ylims = (0.0, 6.0)
    )
end

# Dispatch wrapper providing backward compatibility fallback support for older layout interfaces
function lotka_volterra_sandbox_painter(canvas_idx::Int, θ_physical)
    return lotka_volterra_sandbox_painter(θ_physical)
end

# Generate the complete multi-pane layout frame stream sequence
lv_animation = MinimallyDisruptiveCurves.animate_mdc(
    mdc_curves,
    lotka_volterra_sandbox_painter;
    fps = 20,
    density = 150,
    raw = true
)

# Compile visual frame array and export as a compiled GIF artifact
output_path = joinpath(pwd(), "lotka_volterra_mtk_mdc.gif")
println("Rendering frames and saving video to: $output_path")
Plots.gif(lv_animation, output_path)
