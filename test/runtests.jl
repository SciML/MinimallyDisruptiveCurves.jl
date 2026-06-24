using SafeTestsets
using SciMLTesting

run_tests(;
    core = function ()
        @safetestset "Sparse Initialization Utilities" begin
            include(joinpath(@__DIR__, "test_utilities.jl"))
        end
        @safetestset "Transforms & Pullback Analytics" begin
            include(joinpath(@__DIR__, "test_transforms.jl"))
        end
        @safetestset "Cost Wrapper Mechanics" begin
            include(joinpath(@__DIR__, "test_costs.jl"))
        end
        @safetestset "Solver Pipelines & Integration" begin
            include(joinpath(@__DIR__, "test_solvers.jl"))
        end
        @safetestset "Safety Controls & System Guards" begin
            include(joinpath(@__DIR__, "test_callbacks.jl"))
        end
        @safetestset "Allocation Checks" begin
            include(joinpath(@__DIR__, "alloc_tests.jl"))
        end
        return @safetestset "Explicit Imports Compliance" begin
            include(joinpath(@__DIR__, "explicit_imports.jl"))
        end
    end,
    groups = Dict(
        "Alloc" => joinpath(@__DIR__, "alloc_tests.jl"),
        "ExplicitImports" => joinpath(@__DIR__, "explicit_imports.jl"),
    ),
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    all = ["Core"],
)
