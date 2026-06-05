include("./build_NFKB.jl")


sys = build_nfkb()



# """
#     make_nfkb_cost_function(nfkb_sys; tspan=(0.0, 30000.0), dt=100.0)

# Generates a `CostFunction` instance for MDC tracking using the programmatic MTK structure.
# """
# function make_nfkb_cost_function(nfkb_sys; tspan=(0.0, 30000.0), dt=100.0)
#     # 1. Setup the baseline immutable problem and reference matrix
#     # Turning on the analytic Jacobian gives a massive speedup for stiff solvers
#     base_prob = ODEProblem(nfkb_sys, Dict(), tspan)
    
#     target_times = collect(tspan[1]:dt:tspan[2])
#     nominal_sol = solve(base_prob, Tsit5(); saveat = target_times)
#     target_matrix = Array(nominal_sol)
    
#     # 2. Extract the MTK structural tracking information
#     # This figures out exactly what order MTK expects the raw parameter vector to be in!
#     ps_structure = ModelingToolkit.parameter_values(base_prob)
#     initial_guess, repack, alias = canonicalize(Tunable(), ps_structure)
    
#     # Preallocate a non-allocating Dual cache matching the total parameter count
#     diffcache = DiffCache(copy(initial_guess))
    
#     # 3. Define the objective function f(θ) for MDC
#     # MDC passes raw vectors (θ) directly. This closure accepts Dual or Real types seamlessly.
#     function f(θ)
#         # Grab our type-safe temporary buffer (handles Float64 or Dual numbers flawlessly)
#         buffer = get_tmp(diffcache, θ)
#         copyto!(buffer, θ)
        
#         # Swap the entire parameter array block cleanly using the ultra-fast SciML route
#         ps_updated = replace(Tunable(), ps_structure, buffer)
        
#         # Remake the problem cleanly using the fast parameter object direct route
#         local_prob = remake(base_prob; p = ps_updated)
        
#         sol = solve(local_prob, Tsit5(); saveat = target_times)
        
#         # Catch solver failures from bad parameter combinations
#         if sol.retcode != ReturnCode.Success
#             return Inf
#         end

        
#         current_matrix = Array(sol)
#         return sum((target_matrix .- current_matrix) .^ 2) / length(target_matrix)
#     end
    
#     # 4. Define the exact Automatic Differentiation gradient closure (grad!)
#     function grad!(g, θ)
#         ForwardDiff.gradient!(g, f, θ)
#         return g
#     end
    
#     # Return the clean object, along with the correct initial parameter layout
#     return CostFunction(f, grad!), initial_guess
# end

# println("--- Setting up Biological NFκB MDC Exploration ---")

# # 1. Build the high-performance cost function and extract the matching θ vector
# core_cost, θ_nominal = make_nfkb_cost_function(nfkb)

# # 2. Wire up your standard MDC Transform Chain
# chain = TransformChain(LogAbsTransform()) 
# transformed_cost = TransformedCost(core_cost, chain)

# # @time Hessian_transformed = ForwardDiff.hessian(transformed_cost, θ₀)

# θ₀ = MinimallyDisruptiveCurves.inverse(chain, θ_nominal)
#     dθ₀ =  [-0.0002966819280282935,
#          -0.00046740023865947405,
#          -0.0006172301613469874,
#          -0.0076219915688165345,
#          0.004111126929182724,
#          0.0035317278699176897,
#          -0.0026354024079565993,
#          0.9992471806463734,
#          0.0024844776408676655,
#          0.0006191008133848389,
#          0.0003341884455915324,
#          -9.683482118398813e-5,
#          -0.006320083460264131,
#          -0.0023372269741494464,
#          0.0059844438775523215,
#          0.002022305075160067,
#          0.011612252568162458,
#          0.032650226304473264,
#          0.0004650954874649577,
#          0.0028065194613369642,
#          -0.009986722784235914,
#          -0.0028205754321405825,
#          0.001148377460908286]
# H = 1.0                       # Energy barrier threshold

# sys = MDCSystem(transformed_cost, θ₀, dθ₀, H)

# # 4. Apply your stabilizers and solve
# stabilizer = mdc_momentum_readjustment(sys; tol=1e-3)
# my_pipeline = CallbackSet(stabilizer)

# mdc_curves = MDCsolve(sys, span=MDCSpan(0.0, 5.0), callback=my_pipeline; mode=:fast)
