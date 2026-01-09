using Test

@testset "nopre tests" begin
    @testset "JET Static Analysis" begin
        include("jet_tests.jl")
    end
end
