using ModelingToolkit, Plots
using OrdinaryDiffEq
using ForwardDiff
using PreallocationTools
using SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_values
using SciMLStructures: Tunable, canonicalize, replace
using SciMLBase, LinearAlgebra
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

# A clean, global-safe closure for the value calculation
f_wrapped = θ -> loss_function(θ, p_tuple)

# Pre-allocate the ForwardDiff configuration to keep it allocation-free
x_nominal = getter(prob)
cfg = ForwardDiff.GradientConfig(f_wrapped, x_nominal, ForwardDiff.Chunk(x_nominal))

# An in-place wrapper function that mutates 'g' without modifying package code
grad_wrapped! = function (g, θ)
    ForwardDiff.gradient!(g, f_wrapped, θ, cfg)
end

base_cost = CostFunction(f_wrapped, grad_wrapped!)
pipeline = TransformChain(LogAbsTransform())
final_cost = TransformedCost(base_cost, pipeline)
x_nominal_transformed = MinimallyDisruptiveCurves.inverse(pipeline, x_nominal)
hess0 = ForwardDiff.hessian(θ -> final_cost(θ), x_nominal_transformed)
vs, vals = sparse_eigenbasis(hess0, 5; λ=0.01)

# 1. Initialize an empty dictionary to store the results
mdc_curves = Dict{Int, Any}()

# 2. Loop through the desired indices
for i in 1:5
    println("--- Running MDC for index i = $i ---")
    
    # Create the system dynamically using the i-th direction
    _mdc_sys = MDCSystem(
        final_cost, 
        x_nominal_transformed, 
        vs[i],                # Replaced e_dirs(i) directly with vs[i]
        1.0;                  # Hamiltonian / momentum (H)
        names = params_to_optimize .|> Symbol
    )

    # Set up the pipeline for this iteration
    stabiliser  = mdc_momentum_readjustment(_mdc_sys; tol = 1e-3)
    my_pipeline = CallbackSet(stabiliser)

    # Solve and store the result in your dictionary
    @time curves_i = MDCsolve(_mdc_sys, span = MDCSpan(-10.0, 10.0); callback = my_pipeline)
    
    mdc_curves[i] = curves_i
end
