using JET
using MinimallyDisruptiveCurves
using LinearAlgebra
using Test

"""
JET static analysis tests for MinimallyDisruptiveCurves.jl

These tests verify type stability of core functions using JET.jl,
helping catch type instabilities and potential runtime errors.
"""

# Create test data with non-allocating cost functions
function jet_test_cost(p)
    s = 0.0
    @inbounds for i in eachindex(p)
        s += (p[i] - Float64(i))^2
    end
    return s
end

function jet_test_cost_grad!(p, g)
    s = 0.0
    @inbounds for i in eachindex(p)
        d = p[i] - Float64(i)
        g[i] = 2 * d
        s += d^2
    end
    return s
end

@testset "JET Static Analysis" begin
    # Set up test problem
    p0 = [1.0, 2.0, 3.0]
    dcost = DiffCost(jet_test_cost, jet_test_cost_grad!)
    dp0 = [1.0, 0.0, 0.0]
    dp0 = dp0 / norm(dp0)
    momentum = 100.0
    tspan = (0.0, 1.0)

    mdc = MDCProblem(dcost, p0, dp0, momentum, tspan)

    # Access dynamics function
    f = MinimallyDisruptiveCurves.dynamics(mdc, MinimallyDisruptiveCurves.MDCDynamics())
    du = zeros(6)
    u0 = MinimallyDisruptiveCurves.initial_conditions(mdc)
    g = zeros(3)

    @testset "Dynamics hot path - type stability" begin
        # The dynamics function is the most critical hot path
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) f(
            du, u0, nothing, 0.0
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "Dynamics hot path - no runtime errors" begin
        rep = JET.@report_call target_modules = (MinimallyDisruptiveCurves,) f(
            du, u0, nothing, 0.0
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "DiffCost evaluation - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) dcost(p0)
        @test isempty(JET.get_reports(rep))
    end

    @testset "DiffCost with gradient - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) dcost(p0, g)
        @test isempty(JET.get_reports(rep))
    end

    @testset "initial_costate - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) MinimallyDisruptiveCurves.initial_costate(
            mdc
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "initial_conditions - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) MinimallyDisruptiveCurves.initial_conditions(
            mdc
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "soft_heaviside - type stability" begin
        rep = JET.@report_opt target_modules = (MinimallyDisruptiveCurves,) soft_heaviside(
            0.5, 0.1, 0.3
        )
        @test isempty(JET.get_reports(rep))
    end
end
