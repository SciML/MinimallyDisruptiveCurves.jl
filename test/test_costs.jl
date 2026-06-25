using SafeTestsets

@safetestset "Cost Wrapper Unit Tests" begin
    using MinimallyDisruptiveCurves
    # Import the unexported underlying mechanics
    using MinimallyDisruptiveCurves: forward, inverse, pullback!
    using Test
    using LinearAlgebra

    # ====================================================================
    # --- 1. Define a Mock Cost Function for Testing ---
    # ====================================================================
    # Let's mock a simple Cost Function: C(z) = 0.5 * sum((z .- c).^2)
    # This gives an analytical gradient: dC/dz = z .- c
    struct QuadraticMockCost <: AbstractCost
        center::Vector{Float64}
    end

    # Define the required core API interfaces for MinimallyDisruptiveCurves
    MinimallyDisruptiveCurves.value(cost::QuadraticMockCost, z) = 0.5 * sum((z .- cost.center) .^ 2)

    function MinimallyDisruptiveCurves.gradient!(cost::QuadraticMockCost, grad_buffer, z)
        @. grad_buffer = z - cost.center
        return grad_buffer
    end

    # ====================================================================
    # --- 2. Identity Chain Consistency Verification ---
    # ====================================================================
    @testset "Identity Mapping Baseline" begin
        center = [1.0, 2.0, 3.0]
        raw_cost = QuadraticMockCost(center)

        # Wrap with an empty/identity chain
        tc_identity = TransformedCost(raw_cost, TransformChain())

        θ = [4.0, 5.0, 6.0]
        g_buffer = similar(θ)

        # Value check
        expected_val = MinimallyDisruptiveCurves.value(raw_cost, θ)
        @test tc_identity(θ) ≈ expected_val

        # Value + Gradient functor check
        val_returned = tc_identity(θ, g_buffer)
        @test val_returned ≈ expected_val
        @test g_buffer ≈ (θ - center)
    end

    # ====================================================================
    # --- 3. Composite Sensitivity Map & Finite Differences ---
    # ====================================================================
    @testset "Composite Chain Gradient Accuracy" begin
        center = [2.0, 4.0] # Physical space center target
        raw_cost = QuadraticMockCost(center)

        # Setup a composite transformation pipeline: Scale -> LogAbs
        st = ScaleTransform([2.0, 0.5])
        lat = LogAbsTransform()
        chain = TransformChain(st, lat)

        tc_composite = TransformedCost(raw_cost, chain)

        # Optimization space input parameter configuration
        θ_opt = [1.5, 3.0]
        g_analytical = similar(θ_opt)

        # 1. Compute analytical gradient using the package's pullback mechanics
        val_analytical = tc_composite(θ_opt, g_analytical)

        # 2. Compute numerical gradient via central finite differences (ϵ)
        ϵ = 1.0e-6
        g_numerical = zeros(length(θ_opt))

        for i in 1:length(θ_opt)
            θ_plus = copy(θ_opt)
            θ_minus = copy(θ_opt)

            θ_plus[i] += ϵ
            θ_minus[i] -= ϵ

            val_plus = tc_composite(θ_plus)
            val_minus = tc_composite(θ_minus)

            g_numerical[i] = (val_plus - val_minus) / (2 * ϵ)
        end

        # Test both methods yield identical sensitivities to high numerical precision
        @test val_analytical ≈ tc_composite(θ_opt)
        @test g_analytical ≈ g_numerical rtol = 1.0e-5
    end

# ====================================================================
# --- Combined value_and_gradient! Path ---
# ====================================================================
@testset "value_and_gradient! combined path" begin
    center = [1.0, 2.0, 3.0]
    θ = [4.0, 5.0, 6.0]

    # Counters track how many times the user-side functions are invoked
    sep_f_calls = Ref(0)
    sep_g_calls = Ref(0)
    combined_calls = Ref(0)

    # Separate-call version (matches original v0.4.0 API)
    f_sep = (z) -> begin
        sep_f_calls[] += 1
        return 0.5 * sum((z .- center) .^ 2)
    end
    g_sep! = (g, z) -> begin
        sep_g_calls[] += 1
        @. g = z - center
        return g
    end

    # Combined-call version (shares forward computation)
    fg_combined = (g, z) -> begin
        combined_calls[] += 1
        @. g = z - center
        return 0.5 * sum((z .- center) .^ 2)
    end

    cost_sep = CostFunction(f_sep, g_sep!)
    cost_combined = CostFunction(fg_combined)

    # --- Correctness: same value and gradient ---
    g_buf1 = similar(θ)
    g_buf2 = similar(θ)

    v1 = MinimallyDisruptiveCurves.value_and_gradient!(cost_combined, g_buf1, θ)
    v2 = MinimallyDisruptiveCurves.value_and_gradient!(cost_sep, g_buf2, θ)

    expected_val = 0.5 * sum((θ .- center) .^ 2)
    expected_grad = θ .- center

    @test v1 ≈ v2 ≈ expected_val
    @test g_buf1 ≈ g_buf2 ≈ expected_grad

    # --- Call-count semantics ---
    # Combined path: exactly one user-side call per evaluation
    @test combined_calls[] == 1
    # Separate path: one f call + one g call = two user-side calls
    @test sep_f_calls[] == 1
    @test sep_g_calls[] == 1

    # --- Convenience constructor populates f and g! too ---
    @test MinimallyDisruptiveCurves.value(cost_combined, θ) ≈ expected_val
    g_only = similar(θ)
    MinimallyDisruptiveCurves.gradient!(cost_combined, g_only, θ)
    @test g_only ≈ expected_grad

    # --- Dispatch sanity: confirm the right method is being called ---
    # (use @which to verify the dispatch goes to the expected specialization)
    sep_method = which(MinimallyDisruptiveCurves.value_and_gradient!,
                       (typeof(cost_sep), typeof(g_buf1), typeof(θ)))
    comb_method = which(MinimallyDisruptiveCurves.value_and_gradient!,
                       (typeof(cost_combined), typeof(g_buf2), typeof(θ)))
    @test sep_method !== comb_method  # different specializations
end

@testset "value_and_gradient! flows through TransformedCost" begin
    # Verify the combined path is actually used inside the hot loop, and
    # produces identical results to the separate-call path.
    center = [2.0, 4.0]

    st = ScaleTransform([2.0, 0.5])
    lat = LogAbsTransform()
    chain = TransformChain(st, lat)

    θ_opt = [1.5, 3.0]

    combined_calls = Ref(0)
    fg_combined = (g, z) -> begin
        combined_calls[] += 1
        @. g = z - center
        return 0.5 * sum((z .- center) .^ 2)
    end
    cost_combined = CostFunction(fg_combined)
    tc_combined = TransformedCost(cost_combined, chain)

    sep_f_calls = Ref(0)
    sep_g_calls = Ref(0)
    f_sep = (z) -> begin
        sep_f_calls[] += 1
        return 0.5 * sum((z .- center) .^ 2)
    end
    g_sep! = (g, z) -> begin
        sep_g_calls[] += 1
        @. g = z - center
        return g
    end
    cost_sep = CostFunction(f_sep, g_sep!)
    tc_sep = TransformedCost(cost_sep, chain)

    fwd_caches = MinimallyDisruptiveCurves.generate_fwd_caches(chain, θ_opt)
    N_physical = length(fwd_caches[end])
    gz_buf1 = Vector{Float64}(undef, N_physical)
    gz_buf2 = Vector{Float64}(undef, N_physical)

    g_combined = similar(θ_opt)
    g_sep = similar(θ_opt)

    # Reset counters
    combined_calls[] = 0
    sep_f_calls[] = 0
    sep_g_calls[] = 0

    v_combined = tc_combined(θ_opt, g_combined, gz_buf1, fwd_caches)
    v_sep = tc_sep(θ_opt, g_sep, gz_buf2, fwd_caches)

    @test v_combined ≈ v_sep
    @test g_combined ≈ g_sep

    # Combined: 1 user-side call; Separate: 2 user-side calls
    @test combined_calls[] == 1
    @test sep_f_calls[] == 1
    @test sep_g_calls[] == 1
end
end
