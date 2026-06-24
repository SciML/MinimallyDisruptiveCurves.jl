using MinimallyDisruptiveCurves
using LinearAlgebra
using Test
using AllocCheck
"""
Allocation tests for performance-critical paths in MinimallyDisruptiveCurves.jl
`GROUP="Alloc" julia --project -e 'using Pkg; Pkg.test()`
"""

function test_cost_noalloc(p)
    s = 0.0
    @inbounds for i in eachindex(p)
        s += (p[i] - Float64(i))^2
    end
    return s
end

function test_cost_grad_noalloc!(g, p)
    @inbounds for i in eachindex(p)
        g[i] = 2.0 * (p[i] - Float64(i))
    end
    return nothing
end

@testset "Allocation & Dynamic Hot-Path Tests" begin
    # 1. Build a non-empty chain to force the new zero-allocation pipeline
    chain = TransformChain(ScaleTransform([1.0, 2.0, 1.0]), LogAbsTransform())
    core_cost = CostFunction(test_cost_noalloc, test_cost_grad_noalloc!)
    cost = TransformedCost(core_cost, chain)

    # Define physical space target (where the cost evaluates to zero)
    θ_physical = [1.0, 2.0, 3.0]

    # Elegantly map the physical target back into optimizer space for p0
    p0 = inverse(chain, θ_physical)
    
    # Initial direction in the optimizer space
    dp0 = [0.1, 0.1, 0.1]
    momentum = 10.0

    sys = MDCProblem(cost, p0, dp0, momentum; names = [:a, :b, :c])
    ws = MDCWorkspace(sys)

    λ₀ = MinimallyDisruptiveCurves.initialise_lambda(sys, ws)
    f! = MinimallyDisruptiveCurves.vectorfield(sys)

    u0 = [p0; λ₀]
    du = similar(u0)

    # Warmups
    f!(du, u0, nothing, 0.0)
    MinimallyDisruptiveCurves.mdc_dHdu_residual(sys, u0, 0.0)

    @testset "Dynamics Vector Field Allocations" begin
        allocs = @allocated f!(du, u0, nothing, 0.0)
        @test allocs == 0
    end

    @testset "Dynamics Functional Correctness" begin
        f!(du, u0, nothing, 0.0)
        @test all(isfinite, du)
        @test !all(iszero, du)
    end

    @testset "Mathematical Residual Allocations" begin
        # The 1-arg cost method used here allocates a small temporary array for non-empty chains.
        # We allow this allocation because it runs in the discrete callback, not the ODE hot loop.
        res_allocs = @allocated MinimallyDisruptiveCurves.mdc_dHdu_residual(sys, u0, 1.0)
        @test res_allocs <= 256 
    end

    @testset "TransformedCost Wrap Allocations" begin
        g_buffer = similar(p0)
        
        # Generate the tuple of intermediate buffers using the package's internal helper
        fwd_caches = MinimallyDisruptiveCurves.generate_fwd_caches(chain, p0)
        N_physical = length(fwd_caches[end])
        gz_buffer = Vector{eltype(p0)}(undef, N_physical)

        # Warmups for the 4-arg hot-path used by the ODE vector field
        cost(p0, g_buffer, gz_buffer, fwd_caches)

        # Test the true 4-arg zero-allocation hot-path (Value + Gradient + Buffers)
        @test (@allocated cost(p0, g_buffer, gz_buffer, fwd_caches)) == 0
    end
end
