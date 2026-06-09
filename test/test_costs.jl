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
end
