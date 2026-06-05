using SafeTestsets

@safetestset "Transform Pipeline Unit Tests" begin
    using MinimallyDisruptiveCurves
    using MinimallyDisruptiveCurves: forward, inverse, pullback!, transform_names
    using Test
    using LinearAlgebra

    # ====================================================================
    # --- 1. Empty/Identity Chain Tests ---
    # ====================================================================
    @testset "Empty/Identity Chains" begin
        empty_chain = TransformChain()
        x_test = [1.2, -3.4, 5.6]
        g_test = [1.0, 1.0, 1.0]

        @test forward(empty_chain, x_test) == x_test
        @test inverse(empty_chain, x_test) == x_test
        @test pullback!(empty_chain, g_test, x_test) == g_test
        @test transform_names(empty_chain, [:a, :b]) == [:a, :b]
    end

    # ====================================================================
    # --- 2. Primitive Layers (Isolated Checking) ---
    # ====================================================================
@testset "ScaleTransform" begin
        weights = [2.0, 0.5, 10.0]
        st = ScaleTransform(weights)
        x = [1.0, 4.0, 0.2]
        y = [2.0, 2.0, 2.0]

        # Maps
        @test forward(st, x) ≈ [2.0, 2.0, 2.0]
        @test inverse(st, y) ≈ [1.0, 4.0, 0.2]
        @test inverse(st, forward(st, x)) ≈ x

        # Sensitivity Pullback
        g_out = [1.5, 2.0, -0.5]
        g_in = similar(x)
        pullback!(st, g_in, g_out, x, y)
        @test g_in ≈ [3.0, 1.0, -5.0]

        # Metadata - FIXED: :b is scaled by 0.5, so expect "0.5 * b"
        @test transform_names(st, [:a, :b, :c]) == [Symbol("2.0 * a"), Symbol("0.5 * b"), Symbol("10.0 * c")]
    end

    @testset "LogAbsTransform" begin
        lat = LogAbsTransform()
    
        # x starts in OPTIMIZER SPACE (log domain)
        x = [1.0, 2.0, log(1e-3)]
    
        # 1. Forward Map: Optimizer Space -> Physical Space (exp)
        # exp([1.0, 2.0, log(1e-3)]) -> [exp(1.0), exp(2.0), 1e-3]
        y_expected = [exp(1.0), exp(2.0), 1e-3]
        @test forward(lat, x) ≈ y_expected
    
        # 2. Inverse Map: Physical Space -> Optimizer Space
        @test inverse(lat, forward(lat, x)) ≈ x

        # 3. Sensitivity Pullback
        # y is the output of forward(lat, x) -> [exp(1.0), exp(2.0), 1e-3]
        y = forward(lat, x) 
        g_out = [2.0, 2.0, 2.0] # Gradient coming from physical cost function
        g_in = similar(x)
    
        pullback!(lat, g_in, g_out, x, y)
    
        # Mathematical check: g_in = g_out * y
        @test g_in ≈ [2.0 * exp(1.0), 2.0 * exp(2.0), 2.0 * 1e-3]

        # 4. Metadata
        @test transform_names(lat, [:param]) == [Symbol("log(abs(param))")]
    end
    @testset "FixedParamsTransform" begin
        # 3D parameter space, pinning parameter index 2 to a constant 99.0
        # Free space is 2D -> Maps to 3D Physical Space
        fpt = FixedParamsTransform([1, 3], [99.0], 3)
        
        x_free = [1.5, 4.5]
        y_full = [1.5, 99.0, 4.5]

        # Maps & Boundaries
        @test forward(fpt, x_free) ≈ y_full
        @test inverse(fpt, y_full) ≈ x_free
        @test_throws ErrorException forward(fpt, [1.0, 2.0, 3.0]) # Bad input dimension
        @test_throws ErrorException inverse(fpt, [1.0, 2.0])      # Bad physical dimension

        # Sensitivity Pullback (Slices out active parameter gradients)
        g_out = [10.0, -50.0, 20.0] # -50.0 is on the fixed field
        g_in = similar(x_free)
        pullback!(fpt, g_in, g_out, x_free, y_full)
        @test g_in ≈ [10.0, 20.0]

        # Metadata
        names_3d = [:a, :b, :c]
        @test transform_names(fpt, names_3d) == [:a, :c]
    end

    # ====================================================================
    # --- 3. Compound Transform Chain Pipeline Verification ---
    # ====================================================================
        @testset "Compound Chain Operations" begin
        # Setup an end-to-end composite pipeline:
        # Optimizer space (2D) -> Fix/Mask (3D) -> Scale (3D) -> LogAbs (3D exp) -> Physical Space
    
        free_indices = [1, 3]
        fixed_values = [5.0]
        fpt = FixedParamsTransform(free_indices, fixed_values, 3)
        st  = ScaleTransform([2.0, 1.0, 0.5])
        lat = LogAbsTransform()

        chain = TransformChain(fpt, st, lat)

        x_opt = [2.0, 8.0] # Free parameter starts (in log/optimizer space)
    
        # Manually trace intermediate passes to find analytical target:
        # 1. fpt   -> [2.0, 5.0, 8.0]
        # 2. st    -> [4.0, 5.0, 4.0]               (Multiplies log-space coordinates)
        # 3. lat   -> [exp(4.0), exp(5.0), exp(4.0)] (Forward converts optimizer -> physical via exp)
        expected_y_final = [exp(4.0), exp(5.0), exp(4.0)]

        y_final = forward(chain, x_opt)
        @test y_final ≈ expected_y_final

        # Check inverse reconstruction maps back to optimization space coordinates
        @test inverse(chain, y_final) ≈ x_opt

        # --- Pipeline Pullback Sensitivity Checks ---
        g_initial = [1.0, 1.0, 1.0] # dLoss / dy_final (Gradient sitting at Physical Space)
    
        # Calculate manually backwards (Pullback flows: lat -> st -> fpt):
        # 1. lat pullback: g * y_final => [1.0 * exp(4.0), 1.0 * exp(5.0), 1.0 * exp(4.0)]
        # 2. st pullback:  g * weights => [exp(4.0) * 2.0, exp(5.0) * 1.0, exp(4.0) * 0.5]
        # 3. fpt pullback: slice free  => [2.0 * exp(4.0), 0.5 * exp(4.0)]
        expected_g_transformed = [2.0 * exp(4.0), 0.5 * exp(4.0)]

        g_transformed = pullback!(chain, g_initial, y_final)
        @test g_transformed ≈ expected_g_transformed
    
        # Test structural pipeline names tracing
        opt_names = [:p1, :p2]
        final_names = transform_names(chain, opt_names)
        @test final_names == [Symbol("log(abs(2.0 * p1))"), Symbol("log(abs(p2))")]
    
        # Test full tracking when full physical dimension size names are passed
        physical_names = [:physical_1, :pinned_field, :physical_3]
    
        @test transform_names(chain, physical_names) == [
            Symbol("log(abs(2.0 * physical_1))"),
            Symbol("log(abs(physical_3))")
        ]
    end
end

