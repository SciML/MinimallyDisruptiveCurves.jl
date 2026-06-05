using ModelingToolkit
using OrdinaryDiffEq
using ForwardDiff
using PreallocationTools
using SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_values
using SciMLStructures: Tunable, canonicalize, replace

include("./build_NFKB.jl")
sys = build_nfkb()
tspan = (0.0, 3600.0)
prob = ODEProblem(sys, [], tspan)

timesteps = 0.0:10.0:3600.0
sol_nominal = solve(prob, Tsit5(); saveat = timesteps)
truth_data = Array(sol_nominal(timesteps, idxs = observables(sys)))

function loss_function(x, p_tuple)
    # Destructure optimization context
    odeprob, ts, truth, setter, diffcache, obs_symbols = p_tuple
    
    ps = parameter_values(odeprob)
    buffer = get_tmp(diffcache, x)
    
    # Block-copy baseline tunable values to the dual-safe buffer 
    # (Prevents uninitialized garbage memory if scaling to more parameters)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    
    # Type-safe structural parameter container replacement for ForwardDiff
    ps_updated = replace(Tunable(), ps, buffer)
    
    # Mutate the targeted parameters using the symbolic setter
    setter(ps_updated, x)
    
    # Fast inferred problem recreation
    newprob = remake(odeprob; p = ps_updated)
    sol = solve(newprob, Tsit5(); saveat = ts)
    
    if sol.retcode != SciMLBase.ReturnCode.Success
        return eltype(x)(Inf) # Strict type stability for dual-number propagation
    end
    
    # Extract states cleanly via symbols
    current_data = sol(ts, idxs = obs_symbols)
    
    # Allocation-free MSE calculation
    return sum(abs2, truth .- current_data) / length(truth)
end

params_to_optimize = [
    sys.pathway.a1, 
    sys.pathway.a2
]

setter = setp(prob, params_to_optimize)
getter = getp(prob, params_to_optimize)

raw_ps = parameter_values(prob)
tunable_vector_prototype = copy(canonicalize(Tunable(), raw_ps)[1])
diffcache = DiffCache(tunable_vector_prototype)
obs_symbols = observables(sys)

p_tuple = (prob, timesteps, truth_data, setter, diffcache, obs_symbols)

println("--- Running Loss Function Evaluation ---")

# Extract nominal baseline values cleanly using the new SymbolicIndexingInterface getter
x_nominal = getter(prob)
println("Testing with nominal parameters x = ", x_nominal)

# Evaluate at nominal values
loss_at_nominal = loss_function(x_nominal, p_tuple)
println("Loss at nominal parameters: ", loss_at_nominal)
println("(Expected: ~0.0 or a very small numerical precision error)\n")

# Evaluate at perturbed values (+25%)
x_perturbed = x_nominal .* 1.25  
println("Testing with perturbed parameters x = ", x_perturbed)

loss_at_perturbed = loss_function(x_perturbed, p_tuple)
println("Loss at perturbed parameters: ", loss_at_perturbed)

println("\n--- Testing ForwardDiff Gradient Capability ---")
grad = ForwardDiff.gradient(x -> loss_function(x, p_tuple), x_nominal)
println("Gradient: ", grad)
