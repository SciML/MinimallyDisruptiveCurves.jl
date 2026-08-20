using LinearAlgebra, OrdinaryDiffEq, MinimallyDisruptiveCurves, Plots, ForwardDiff, SymbolicIndexingInterface
include("./cost_functions.jl")

H = 1.0              # Parameter exploration kinetic energy threshold


# ====================================================================
# --- Mass Spring Setup ---
# ====================================================================

# Define our base physical system nominal profile
θ₀ = [1.0, 2.0, 2.0]
u0_physical = [1.0, 0.0]
tspan_physical = (0.0, 10.0)

ms_cost = make_mass_spring_mse_cost_function(θ₀, u0 = u0_physical, tspan = tspan_physical)

# Alternative initial conditions
perfect_initial_direction = [1.0, 2.0, 2.0]
good_initial_direction = [2.0, 2.0, 2.0] # angle of approximately 16° from the ideal angle
poor_initial_direction = [1.0, 0.0, 0.0] # angle of approximately 48° from the ideal angle
terrible_initial_direction = [-1.0, 1.0, 0.0] # angle of approximately 76° from the ideal angle

ms_mdc = MDCProblem(ms_cost, θ₀, good_initial_direction, H; names = [:mass, :damping, :stiffness])

# ====================================================================
# --- Sin cost setup ---
# ====================================================================
A = 1
ω = 1

sin_cost = make_sin_cost_function(A, ω)

sin_initial_direction = [1.0,1.0]

sin_mdc = MDCProblem(sin_cost, [0.0, 0.0], sin_initial_direction, H; names = [:x, :y])

# ====================================================================
# --- Lotka Volterra Setup ---
# ====================================================================

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p_nominal = [1.5, 1.0, 3.0, 1.0]
lv_cost = make_lv_cost_function(p_nominal, u0, tspan)

hess0 = ForwardDiff.hessian(lv_cost.f, p_nominal)
eigen_decomposition = eigen(hess0)
init_dir = eigen_decomposition.vectors[:, 1]

lv_mdc = MDCProblem(
    lv_cost,
    p_nominal,
    init_dir,
    H;                
    names = [:α, :β, :δ, :γ]
)

# ====================================================================
# --- NFKB Setup ---
# ====================================================================

include("./build_NFKB.jl")
sys = build_nfkb()
tspan = (0.0, 3600.0)
prob = ODEProblem(sys, [], tspan)

target_observables = [
    sys.pathway.NFkBn_obs,
    sys.pathway.IkBa_cyto_obs,
    sys.pathway.A20t_obs,
    sys.pathway.IKKtot_obs,
    sys.pathway.IKKa_obs,
    sys.pathway.IkBat_obs,
]

params_to_optimize = tunable_parameters(sys) ∩ parameters(sys)

# ==========================================
# 4. Generate Baseline Experimental Data ("Truth")
# ==========================================
timesteps = 0.0:10.0:3600.0
sol_nominal = solve(prob, Tsit5(); saveat = timesteps)
truth_data = Array(sol_nominal(timesteps, idxs = target_observables))

# ==========================================
# 6. Build the Optimization Context
# ==========================================
setter = setp(prob, params_to_optimize)
getter = getp(prob, params_to_optimize)

raw_ps = parameter_values(prob)
tunable_vector_prototype = copy(canonicalize(Tunable(), raw_ps)[1])
diffcache = DiffCache(tunable_vector_prototype)

# Package context containing our 24 target parameters and 6 observables
p_tuple = (prob, timesteps, truth_data, setter, diffcache, target_observables)

# ==========================================
# 7. Evaluation and Gradient Verification
# ==========================================

x_nominal = getter(prob)
loss_at_nominal = nfkb_loss_function(x_nominal, p_tuple)

# A clean, global-safe closure for the value calculation
f_wrapped = θ -> nfkb_loss_function(θ, p_tuple)

# Pre-allocate the ForwardDiff configuration to keep it allocation-free
x_nominal = getter(prob)
cfg = ForwardDiff.GradientConfig(f_wrapped, x_nominal, ForwardDiff.Chunk(x_nominal))

# An in-place wrapper function that mutates 'g' without modifying package code
grad_wrapped! = function (g, θ)
    return ForwardDiff.gradient!(g, f_wrapped, θ, cfg)
end

base_cost = CostFunction(f_wrapped, grad_wrapped!)
pipeline = TransformChain(LogAbsTransform())
final_cost = TransformedCost(base_cost, pipeline)

x_nominal_transformed = MinimallyDisruptiveCurves.inverse(pipeline, x_nominal)
hess0 = ForwardDiff.hessian(θ -> final_cost(θ), x_nominal_transformed)
vs, vals = sparse_eigenbasis(hess0, 5; λ = 0.01)

nfkb_mdc = MDCProblem(
        final_cost,
        x_nominal_transformed,
        vs[2],
        H;
        names = params_to_optimize .|> Symbol
    )

# ====================================================================
# --- MDC Setup ---
# ====================================================================

# Change this to select which system should be tested
sys = sin_mdc

stabilizer = mdc_momentum_readjustment(sys; tol = 1.0e-3)
logger = mdc_verbose_callbacks(sys, range(-2.0, 5.0, 20))

my_pipeline = CallbackSet(stabilizer, logger...)
log_pipeline = CallbackSet(logger...)
pipeline_to_use = log_pipeline

span = MDCSpan(0.0, 10.0)

# select the running mode of MDC solve, options are :adaptive, :fast, :fixed
mode = :adaptive

# ====================================================================
# --- PI Setup ---
# ====================================================================

use_pi_control = true
min_exp = -4.0
K_length = 8


# K_values selects which PI params should be tested, the same value set is used for Kp and Ki values so the run time grows with the square of the length of K_values
# K_values = [0.0, 10.0 .^ range(min_exp, 0.0, length = K_length - 1)...]
K_values = [0.0, 1e-4, 1e-3, 0.01, 0.1, 0.3, 0.6, 1.0]


results = [MDCSolve(sys, span = span, callback = pipeline_to_use, use_pi_control = use_pi_control, pi_params = [Kp, Ki], mode = mode).positive_sol.u[end][end]
         for Kp in K_values, Ki in K_values]

grid_kp = [i for i in 1:length(K_values) for j in 1:length(K_values)]
grid_ki = [j for i in 1:length(K_values) for j in 1:length(K_values)]

costs = vec(results)

print(results)

result_plot = scatter(
    grid_kp,
    grid_ki,
    marker_z = costs,         # Map point color to total_cost
    xlabel = "Kp",
    ylabel = "Ki",
    xticks = (1:length(K_values), K_values),
    yticks = (1:length(K_values), K_values),
    title = "Parameter Evaluations",
    colorbar = true,
    colorbar_title = "Total Cost",
    color = :viridis,          # Colormap: :viridis, :turbo, :cividis, etc.
    markersize = 5,            # Size of individual dots
    markerstrokewidth = 0.5,   # Add subtle outline around markers
    markerstrokecolor = :black,
    legend = false
)

savefig(result_plot, "plots/mdc_sin.png")