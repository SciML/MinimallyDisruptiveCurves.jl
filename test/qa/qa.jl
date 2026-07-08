using SciMLTesting
using JET
using MinimallyDisruptiveCurves
using Test

"""
JET static analysis tests for MinimallyDisruptiveCurves.jl

These tests verify type stability and check for runtime errors of core functions
using JET.jl, matching the updated package API architecture.
"""

# Create non-allocating test cost functions matching precompilation workflow
function jet_test_cost(p)
    return (1.0 - p[1])^2 + 10.0 * (p[2] - p[1]^2)^2
end

function jet_test_cost_grad!(g, p)
    ε = 1.0e-8
    p2 = copy(p)

    for i in eachindex(p)
        orig = p2[i]
        p2[i] = orig + ε
        g[i] = (jet_test_cost(p2) - jet_test_cost(p)) / ε
        p2[i] = orig
    end

    return nothing
end

@testset "JET Static Analysis" begin
    # --------------------------------------------------------------------------
    # Setup Test Data & System Architecture
    # --------------------------------------------------------------------------
    θ_phys = [1.1, 1.1]
    dθ_phys = [1.2, 1.2]

    cost = CostFunction(jet_test_cost, jet_test_cost_grad!)

    chain = TransformChain(
        ScaleTransform([1.0, 1.0]),
        LogAbsTransform()
    )

    tcost = TransformedCost(cost, chain)

    θ₀ = inverse(chain, θ_phys)
    dθ₀ = inverse(chain, dθ_phys)

    sys = MDCProblem(
        tcost,
        θ₀,
        dθ₀,
        10.0;
        names = [:A, :B]
    )

    ws = MinimallyDisruptiveCurves.MDCWorkspace(sys)

    # Pre-allocate containers for vector field execution
    vf! = MinimallyDisruptiveCurves.vectorfield(sys)
    λ0 = MinimallyDisruptiveCurves.initialise_lambda(sys, ws)
    u = vcat(θ₀, λ0)
    du = similar(u)

    # --------------------------------------------------------------------------
    # JET Analysis Tests
    # --------------------------------------------------------------------------
    @testset "Cost evaluation - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) tcost(θ₀)
        @test isempty(JET.get_reports(rep))
    end

    @testset "Cost + gradients - type stability" begin
        g = similar(θ₀)
        gz = similar(θ_phys)
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) tcost(θ₀, g, gz)
        @test isempty(JET.get_reports(rep))
    end

    @testset "Lambda initialisation - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) MinimallyDisruptiveCurves.initialise_lambda(sys, ws)
        @test isempty(JET.get_reports(rep))
    end

    @testset "Vector field factory - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) MinimallyDisruptiveCurves.vectorfield(sys)
        @test isempty(JET.get_reports(rep))
    end

    @testset "Vector field execution - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) vf!(du, u, nothing, 0.0)
        @test isempty(JET.get_reports(rep))
    end

    @testset "Verbose Callbacks - no runtime errors" begin
        # This will statically catch unbound variables (like the missing N)
        # without needing to run a full ODE solve.
        rep = JET.@report_call target_modules = (MinimallyDisruptiveCurves,) mdc_verbose_callbacks(sys, [0.1, 0.2])
        @test isempty(JET.get_reports(rep))
    end


    @testset "Vector field execution - no runtime errors" begin
        rep = JET.@report_call target_modules = (MinimallyDisruptiveCurves,) vf!(du, u, nothing, 0.0)
        @test isempty(JET.get_reports(rep))
    end
end
run_qa(
    MinimallyDisruptiveCurves;
    explicit_imports = true,
)
