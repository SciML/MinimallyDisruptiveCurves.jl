# Precompilation workload for MinimallyDisruptiveCurves.jl
# This file is included at the end of the main module to cache compiled methods.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # ----------------------------------------------------------------
    # 1. Setup Mock Data and Mock Functions
    # ----------------------------------------------------------------
    # A simple 2D Rosenbrock-like cost function for execution speed
    function _simple_cost(p)
        return (1.0 - p[1])^2 + 10.0 * (p[2] - p[1]^2)^2
    end

    # Allocation-free finite-difference gradient loop
    function _simple_cost_grad!(g, p)
        ε = 1.0e-8
        # Cache a single mutable working vector instead of recreating it in the loop
        p_perturbed = copy(p)
        for i in eachindex(p)
            orig = p_perturbed[i]
            p_perturbed[i] = orig + ε
            cost_plus = _simple_cost(p_perturbed)
            p_perturbed[i] = orig # Reset element

            g[i] = (cost_plus - _simple_cost(p)) / ε
        end
        return nothing
    end

    # Define physical baseline configurations
    θ_physical_nominal = [1.1, 1.1]
    dθ_physical_nominal = [1.2, 1.2]
    H_val = 10.0         # Ensure H > initial cost
    mock_physical_names = [:param_A, :param_B]

    @compile_workload begin
        # ----------------------------------------------------------------
        # 2. Precompile Cost and Transform Chains
        # ----------------------------------------------------------------
        core_cost = CostFunction(_simple_cost, _simple_cost_grad!)

        # Build a complete mock transform chain (Scale -> LogAbs)
        w_vec = [1.0, 1.0]
        chain = TransformChain(ScaleTransform(w_vec), LogAbsTransform())

        # Wrap into your TransformedCost structure
        t_cost = TransformedCost(core_cost, chain)

        # Map physical test parameters into the internal optimization space coordinates
        θ₀ = MinimallyDisruptiveCurves.inverse(chain, θ_physical_nominal)
        dθ₀ = MinimallyDisruptiveCurves.inverse(chain, dθ_physical_nominal)

        # Determine the length of the physical space vector
        N_physical = length(MinimallyDisruptiveCurves.forward(chain, θ₀))

        # Run mock evaluations with the updated, allocation-free 3-argument signature
        _val = t_cost(θ₀)
        g_buffer = similar(θ₀)
        gz_buffer = Vector{eltype(θ₀)}(undef, N_physical)

        # Caches the new zero-allocation functor pathway
        _val_grad = t_cost(θ₀, g_buffer, gz_buffer)

        # ----------------------------------------------------------------
        # 3. Precompile System & Workspace Initialization
        # ----------------------------------------------------------------
        sys = MDCProblem(t_cost, θ₀, dθ₀, H_val; names = mock_physical_names)
        ws = MDCWorkspace(sys)

        # Trigger internal factory and lambda allocations
        _λ₀ = MinimallyDisruptiveCurves.initialise_lambda(sys, ws)
        _vf! = MinimallyDisruptiveCurves.vectorfield(sys)

        # ----------------------------------------------------------------
        # 4. Precompile Solvers and Callbacks (The heaviest step)
        # ----------------------------------------------------------------
        # Use a minuscule span limit to keep precompilation execution near-instant
        tiny_span = MDCSpan(-0.01, 0.01)

        # Setup the default safety callback
        cb_safety = mdc_safety_callback(sys)

        # Solve the ODE (compiles Tsit5, OrdinaryDiffEq routines, and your vector field)
        curve = MDCSolve(sys; span = tiny_span, callback = cb_safety)

        # ----------------------------------------------------------------
        # 5. Precompile Interpolation and Base Extensions
        # ----------------------------------------------------------------
        if !isnothing(curve.positive_sol) || !isnothing(curve.negative_sol)
            # Trace interpolation paths
            _ = curve(0.0, type = :all)
            _ = curve(0.0, type = :parameters)
            _ = curve(0.0, type = :costates)

            # Precompile the custom string printing layouts
            show_buffer = IOBuffer()
            show(show_buffer, MIME"text/plain"(), curve)
        end
    end
end
