using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(path = joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    include(joinpath(@__DIR__, "qa", "qa.jl"))
else
    @testset "MinimallyDisruptiveCurves.jl Suite" begin

        # --- 1. CORE FUNCTIONAL TESTS ---
        if GROUP == "All" || GROUP == "Core"
            @safetestset "Transforms & Pullback Analytics" begin
                include("test_transforms.jl")
            end

            @safetestset "Cost Wrapper Mechanics" begin
                include("test_costs.jl")
            end

            @safetestset "Solver Pipelines & Integration" begin
                include("test_solvers.jl")
            end

            @safetestset "Safety Controls & System Guards" begin
                include("test_callbacks.jl")
            end
        end

        # --- 2. PERFORMANCE ALLOCATION CHECKING ---
        if GROUP == "All" || GROUP == "Core" || GROUP == "Alloc"
            @safetestset "Allocation Checks" begin
                include("alloc_tests.jl") # Keeps your allocation tests isolated
            end
        end

        # --- 3. LINTING AND IMPORT CLEANLINESS ---
        if GROUP == "All" || GROUP == "Core" || GROUP == "ExplicitImports"
            @safetestset "Explicit Imports Compliance" begin
                include("explicit_imports.jl") # Verifies scoping rules
            end
        end

    end
end
