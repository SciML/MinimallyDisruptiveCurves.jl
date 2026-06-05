using ModelingToolkit
using OrdinaryDiffEq
using ForwardDiff
using PreallocationTools
using SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_values
using SciMLStructures: Tunable, canonicalize, replace
using SciMLBase
# ==========================================
# 1. Setup MTK Network & Base ODE Problem
# ==========================================
include("./build_NFKB.jl")
sys = build_nfkb()
tspan = (0.0, 3600.0)
prob = ODEProblem(sys, [], tspan)

# ==========================================
# 2. Define the Specific Tracking States (Observables)
# ==========================================
target_observables = [
    sys.pathway.NFkBn_obs,
    sys.pathway.IkBa_cyto_obs,
    sys.pathway.A20t_obs,
    sys.pathway.IKKtot_obs,
    sys.pathway.IKKa_obs,
    sys.pathway.IkBat_obs
]

# ==========================================
# 3. Dynamic Parameter Identification
# ==========================================
# Clean intersection ensures we only pass active, system-level symbols
params_to_optimize = tunable_parameters(sys) ∩ parameters(sys)

println("Natively optimizing $(length(params_to_optimize)) tunable parameters.")

# ==========================================
# 4. Generate Baseline Experimental Data ("Truth")
# ==========================================
timesteps = 0.0:10.0:3600.0
sol_nominal = solve(prob, Tsit5(); saveat = timesteps)
truth_data = Array(sol_nominal(timesteps, idxs = target_observables))

# ==========================================
# 5. High-Performance Loss Function
# ==========================================
function loss_function(x, p_tuple)
    # Destructure context tuple
    odeprob, ts, truth, setter, diffcache, obs_symbols = p_tuple
    
    ps = parameter_values(odeprob)
    buffer = get_tmp(diffcache, x)
    
    # Block-copy baseline values (the non-tunables stay untouched elsewhere in `ps`)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    
    # Type-safe structural parameter container replacement for ForwardDiff
    ps_updated = replace(Tunable(), ps, buffer)
    
    # Mutate only our active dual/float optimization array
    setter(ps_updated, x)
    
    # Fast inferred problem recreation
    newprob = remake(odeprob; p = ps_updated)
    sol = solve(newprob, Tsit5(); saveat = ts)
    
    if sol.retcode != SciMLBase.ReturnCode.Success
        return eltype(x)(Inf) # Strict type stability for dual-number propagation
    end
    
    # Extract states cleanly via targeted tracking symbols
    current_data = sol(ts, idxs = obs_symbols)
    
    # Allocation-free MSE over the exact matrix of specified states
    return sum(abs2, truth .- current_data) / length(truth)
end

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
println("\n--- Running Scaled Loss Function Evaluation ---")

x_nominal = getter(prob)
loss_at_nominal = loss_function(x_nominal, p_tuple)
println("Loss at nominal parameters: ", loss_at_nominal)

# Perturb all 24 parameters at once to check scaling response
x_perturbed = x_nominal .* 1.10  
loss_at_perturbed = loss_function(x_perturbed, p_tuple)


# A clean, global-safe closure for the value calculation
f_wrapped = θ -> loss_function(θ, p_tuple)

# Pre-allocate the ForwardDiff configuration to keep it lightning-fast and allocation-free
x_nominal = getter(prob)
cfg = ForwardDiff.GradientConfig(f_wrapped, x_nominal, ForwardDiff.Chunk(x_nominal))

# An in-place wrapper function that mutates 'g' without modifying package code
grad_wrapped! = function (g, θ)
    ForwardDiff.gradient!(g, f_wrapped, θ, cfg)
end


# This instantiates your package structs perfectly without a single line changed inside it!
base_cost = CostFunction(f_wrapped, grad_wrapped!)

pipeline = TransformChain(LogAbsTransform())
final_cost = TransformedCost(base_cost, pipeline)
