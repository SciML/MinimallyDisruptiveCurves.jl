# ====================================================================
# MINIMALLY DISRUPTIVE CURVES (MDC) — LOTKA-VOLTERRA MANIFOLD TRACE
# ====================================================================

using LinearAlgebra
using OrdinaryDiffEq
using MinimallyDisruptiveCurves
using Statistics
using Plots
using ForwardDiff
using Optimization  # Brought in for OptimizationFunction
using ModelingToolkit
import ModelingToolkit: SymbolicT
import SciMLBase: successful_retcode
import ModelingToolkit.t_nounits
import ModelingToolkit.D_nounits
using Printf

# Global Configuration Parameters
const TSPAN_PHYSICAL = (0.0, 10.0)  # Time horizon for the differential equations
const EXPLORATION_DIR = 2           # Hessian eigenvector index selected for manifold path initialization

# ====================================================================
# 1. MODEL DEFINITION via ModelingToolkit (MTK)
# ====================================================================

@component function LotkaVolterra(;
        name, α = 1.3, β = 0.9, γ = 0.8, δ = 1.8)
    @parameters begin
        α = α
        β = β
        γ = γ
        δ = δ
    end
    params = SymbolicT[]
    push!(params, α); push!(params, β); push!(params, γ); push!(params, δ)

    @variables begin
        x(t_nounits)
        y(t_nounits)
    end
    vars = SymbolicT[]
    push!(vars, x); push!(vars, y)

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

# ====================================================================
# 2. RAW OPTIMIZATION FUNCTION & BASELINE EXTRACTION
# ====================================================================
println("--- Extracting MTK Structural Layouts ---")

# Map MTK model to a simulation that depends on a flat parameter vector.
mapping = mtk_parameter_mapping(sys; tspan = TSPAN_PHYSICAL)
p_nominal = mapping.θ_nominal

# Precompute target baseline features using nominal trajectory mapping
nom_sol = mapping.simulator(p_nominal)
const grid_steps = range(TSPAN_PHYSICAL[1], TSPAN_PHYSICAL[2], length = 200)

# Precompute target traits exactly as numbers
const nom_features = [
    mean(nom_sol(t_val)[1] for t_val in grid_steps),
    maximum(nom_sol(t_val)[2] for t_val in grid_steps)
]

"""
    loss_function(p, __args...)

Pure-numeric loss function mapping raw parameter vector `p` directly into 
the SciML simulator and calculating feature disruption metrics.
"""
function loss_function(p, __args...)
    sol = mapping.simulator(p)
    
    # Validation penalty wall
    if sol.retcode != ReturnCode.Success && sol.retcode != ReturnCode.Default
        return 10000.0
    end

    # Explicit array tracking loop completely safe for ForwardDiff
    vals = [sol(t_val) for t_val in grid_steps]
    
    if any(any(isnan.(v)) || any(isinf.(v)) for v in vals)
        return 10000.0
    end

    mean_prey    = mean(v[1] for v in vals)
    max_predator = maximum(v[2] for v in vals)
    
    return sum(abs2, [mean_prey, max_predator] .- nom_features)
end

# Build the explicit OptimizationFunction layout manually with ForwardDiff AD
opt_f = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())

# Wrap into the lightweight .f layout structure expected by MDCSystem
core_cost = (f = p -> opt_f(p, nothing),)

# ====================================================================
# 3. MDC GENERATION PIPELINE
# ====================================================================
println("--- Setting up MTK Lotka-Volterra MDC System ---")

# Compute the local curvature Hessian matrix manually
hess0 = ForwardDiff.hessian(core_cost.f, p_nominal)
eigen_decomposition = eigen(Symmetric(hess0))

# Extract selected eigenvector corresponding to initial curve direction
init_dir = eigen_decomposition.vectors[:, EXPLORATION_DIR]

# Extract parameter names cleanly from the system parameters metadata vector
param_names = [Symbol(Symbolics.getname(p)) for p in parameters(sys)]

# Build MDC system using explicit names array 
mdc_sys = MDCSystem(
    core_cost, 
    p_nominal, 
    init_dir, 
    1.0;                 # Hamiltonian / momentum (H)
    names = param_names  # Explicitly matches mapping symbols: [:α, :β, :γ, :δ]
)

# Attach conservation stabilization callback routine to mitigate integration drift
stabiliser  = mdc_momentum_readjustment(mdc_sys; tol = 1e-3)
my_pipeline = CallbackSet(stabiliser)

println("Launching MDC...")
@time mdc_curves = MDCsolve(mdc_sys, span = MDCSpan(-5.0, 5.0); callback = my_pipeline)

# ====================================================================
# 4. CUSTOM VISUALIZATION AND ANIMATION ENGINE INTERACTION
# ====================================================================
println("\nPreparing continuous manifold animation...")

function lotka_volterra_sandbox_painter(θ_physical)
    plot_t_grid = range(TSPAN_PHYSICAL[1], TSPAN_PHYSICAL[2], length = 200)
    
    sol_nominal   = mapping.simulator(p_nominal)
    sol_perturbed = mapping.simulator(θ_physical)
    
    states_nom  = [sol_nominal(t_val) for t_val in plot_t_grid]
    states_pert = [sol_perturbed(t_val) for t_val in plot_t_grid]
    
    prey_nominal = [u[1] for u in states_nom]
    pred_nominal = [u[2] for u in states_nom]
    
    prey_perturbed = [u[1] for u in states_pert]
    pred_perturbed = [u[2] for u in states_pert]

    mean_prey_nom  = mean(prey_nominal)
    mean_prey_pert = mean(prey_perturbed)
    max_pred_nom   = maximum(pred_nominal)
    max_pred_pert  = maximum(pred_perturbed)

    Plots.plot!(
        plot_t_grid, [prey_nominal pred_nominal], 
        subplot = 1, 
        linealpha = 0.20, linestyle = :dash, 
        color = [:blue :red], label = false
    )
    
    Plots.plot!(
        plot_t_grid, [prey_perturbed pred_perturbed], 
        subplot = 1, 
        linewidth = 2, 
        color = [:blue :red], label = ["Prey (x)" "Predator (y)"],
        legend = :topright
    )
    
    Plots.hline!([mean_prey_nom], subplot = 1, linestyle = :dot, linealpha = 0.4, color = :blue, label = false)
    Plots.hline!([max_pred_nom], subplot = 1, linestyle = :dot, linealpha = 0.4, color = :red, label = false)
    
    Plots.hline!([mean_prey_pert], subplot = 1, linestyle = :dashdot, linewidth = 1.2, color = :darkblue, label = "Mean Prey")
    Plots.hline!([max_pred_pert], subplot = 1, linestyle = :dashdot, linewidth = 1.2, color = :darkred, label = "Max Predator")
    
    Plots.plot!(
        subplot = 1,
        xlabel = "Time", ylabel = "Population",
        xlims = TSPAN_PHYSICAL,
        ylims = (0.0, 6.0)
    )
end

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

# Export as GIF
output_path = joinpath(pwd(), "lotka_volterra_mtk_mdc.gif")
println("Rendering frames and saving video to: $output_path")
Plots.gif(lv_animation, output_path)
