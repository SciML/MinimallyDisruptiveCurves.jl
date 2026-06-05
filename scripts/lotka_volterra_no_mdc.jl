using LinearAlgebra, OrdinaryDiffEq, MinimallyDisruptiveCurves, Statistics, Plots, ForwardDiff, LaTeXStrings

# Which eigenvector (direction) to explore along the manifold
which_dir = 1

# ====================================================================
# --- Core Physics Engine (Lotka-Volterra) ---
# ====================================================================

function lotka_volterra_dynamics!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2] 
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2] 
    return nothing
end

u0 = [1.0, 1.0] 
tspan = (0.0, 10.0) 
p_nominal = [1.5, 1.0, 3.0, 1.0] 

nom_prob = ODEProblem(lotka_volterra_dynamics!, u0, tspan, p_nominal)

# Helper for evaluations
solve_at_p(p) = solve(remake(nom_prob; p=p), Tsit5())

# ====================================================================
# --- Objective Features & Cost Framework ---
# ====================================================================

function extract_features(p)
    sol = solve_at_p(p)
    grid = range(0.0, 10.0, length=200)
    
    # Efficient lazy evaluations utilizing solution interpolation
    mean_prey = mean(sol(t)[1] for t in grid)
    max_predator = maximum(sol(t)[2] for t in grid)
    
    return [mean_prey, max_predator]
end

nom_features = extract_features(p_nominal)

function loss(p)
    p_features = extract_features(p)
    return sum(abs2, p_features .- nom_features)
end

function loss_grad!(g, p)
    ForwardDiff.gradient!(g, loss, p)
    return g
end

core_cost = CostFunction(loss, loss_grad!)

# ====================================================================
# --- Execution Pipeline ---
# ====================================================================

println("--- Setting up Lotka-Volterra MDC System ---")

# Compute the local Hessian at nominal parameters to discover insensitive directions
hess0 = ForwardDiff.hessian(loss, p_nominal)
eigen_decomposition = eigen(hess0)
init_dir = eigen_decomposition.vectors[:, which_dir]

# Leverage our automated IdentityTransform fallbacks to keep the top layer clean
sys = MDCSystem(
    core_cost, 
    p_nominal, 
    init_dir, 
    1.0;                  # Energy Headroom (H)
    names=[:p₁, :p₂, :p₃, :p₄]
)

println("Launching parallel manifold integration...")
mdc_curves = MDCsolve(sys, span=MDCSpan(-1.0, 5.0))

# ====================================================================
# --- Animation Pipeline Integration ---
# ====================================================================

println("\nPreparing continuous manifold animation...")

# Define the live sandbox rendering function
function lotka_volterra_sandbox_painter(θ_physical)
    plot_t_grid = range(tspan[1], tspan[2], length = 200)
    
    sol_nominal   = solve_at_p(p_nominal)
    sol_perturbed = solve_at_p(θ_physical)
    
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
        xlims = tspan,
        ylims = (0.0, 6.0)
    )
end

# Invoke the updated linear animation tracking utility function
lv_animation = MinimallyDisruptiveCurves.animate_mdc(
    mdc_curves,
    lotka_volterra_sandbox_painter;
    fps = 20,
    density = 150,  
    raw = true      
)

output_path = joinpath(pwd(), "lotka_volterra2_mdc.gif")
println("Rendering frames and saving video to: $output_path")
Plots.gif(lv_animation, output_path)
