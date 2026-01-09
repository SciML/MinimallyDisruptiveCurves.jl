using MinimallyDisruptiveCurves
using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "MinimallyDisruptiveCurves.jl" begin
    if GROUP == "All" || GROUP == "Core"
        @testset "mass_spring" begin
            include("mass_spring.jl")
        end
    end

    if GROUP == "All" || GROUP == "Alloc"
        @testset "allocation_tests" begin
            include("alloc_tests.jl")
        end
    end

    if GROUP == "All" || GROUP == "ExplicitImports"
        @testset "Explicit Imports" begin
            include("explicit_imports.jl")
        end
    end
end
