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
    # 2. Build new structural pipeline components
    core_cost = CostFunction(test_cost_noalloc, test_cost_grad_noalloc!)
    cost      = TransformedCost(core_cost) # Identity transform chain default
    
    p0       = [1.0, 2.0, 3.0]
    dp0      = [1.0, 0.0, 0.0]
    momentum = 10.0 

    sys = MDCSystem(cost, p0, dp0, momentum; names = [:a, :b, :c])
    ws  = MDCWorkspace(sys)

    λ₀ = MinimallyDisruptiveCurves.initialise_lambda(sys, ws)
    f! = vectorfield(sys)

    u0 = [p0; λ₀]
    du = similar(u0)

    # Warmups
    f!(du, u0, nothing, 0.0)
    MinimallyDisruptiveCurves.mdc_dHdu_residual(sys, u0, 0.0)

    @testset "Dynamics Vector Field Allocations" begin
        allocs = @allocated f!(du, u0, nothing, 0.0)
        @test allocs  < 100 # sacrificed completely allocation free as cost function evaluation is the major cost.allocs=80 for current version
    end

    @testset "Dynamics Functional Correctness" begin
        f!(du, u0, nothing, 0.0)
        @test all(isfinite, du)
        @test !all(iszero, du) 
    end

    @testset "Mathematical Residual Allocations" begin
        res_allocs = @allocated MinimallyDisruptiveCurves.mdc_dHdu_residual(sys, u0, 1.0)
        @test res_allocs == 0
    end

    @testset "TransformedCost Wrap Allocations" begin
        g_buffer = similar(p0)
        
        # Warmups - Matching your (θ, gθ) functor interface
        cost(p0)
        cost(p0, g_buffer)

        # Test value-only pass
        @test (@allocated cost(p0)) < 400

        # Test value + gradient pullback pass (θ first, gθ second)
        @test (@allocated cost(p0, g_buffer)) <800
    end



end
