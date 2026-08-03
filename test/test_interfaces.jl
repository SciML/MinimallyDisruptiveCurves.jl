using MinimallyDisruptiveCurves
using Test

import MinimallyDisruptiveCurves: forward, inverse, pullback!, transform_names, value, gradient!

struct ShiftTransform <: AbstractTransform
    shift::Float64
    pullback_aliases_input::Base.RefValue{Bool}
end

forward(transform::ShiftTransform, x) = x .+ transform.shift
inverse(transform::ShiftTransform, y) = y .- transform.shift

function pullback!(transform::ShiftTransform, g_in, g_out, x, y)
    transform.pullback_aliases_input[] = g_in === x || g_in === y
    copyto!(g_in, g_out)
    return g_in
end

struct QuadraticCost <: AbstractCost
    center::Vector{Float64}
end

value(cost::QuadraticCost, z) = sum(abs2, z .- cost.center) / 2

function gradient!(cost::QuadraticCost, g, z)
    @. g = z - cost.center
    return g
end

@testset "AbstractTransform extension" begin
    pullback_aliases_input = Ref(false)
    transform = ShiftTransform(2.0, pullback_aliases_input)
    x = [1.0, -3.0]
    y = forward(transform, x)
    @test y == [3.0, -1.0]
    @test inverse(transform, y) == x

    g_out = [4.0, -5.0]
    g_in = similar(g_out)
    @test pullback!(transform, g_in, g_out, x, y) === g_in
    @test g_in == g_out

    chain = TransformChain(transform, ScaleTransform([3.0, 0.5]))
    @test transform_names(chain, [:x, :y]) == [Symbol("3.0 * x"), Symbol("0.5 * y")]
    buffers = generate_fwd_caches(chain, x)
    @test forward!(chain, buffers, x) == [9.0, -0.5]

    chain_gradient = similar(x)
    @test pullback!(chain, chain_gradient, [2.0, 4.0], buffers) === chain_gradient
    @test chain_gradient == [6.0, 2.0]
    @test !pullback_aliases_input[]
end

@testset "AbstractCost extension" begin
    cost = QuadraticCost([1.0, -1.0])
    z = [4.0, 3.0]
    gradient = similar(z)

    @test value(cost, z) == 12.5
    @test value_and_gradient!(cost, gradient, z) == 12.5
    @test gradient == [3.0, 4.0]

    transformed = TransformedCost(cost, TransformChain(ShiftTransform(1.0, Ref(false))))
    transformed_gradient = similar(z)
    @test transformed(z, transformed_gradient) == 20.5
    @test transformed_gradient == [4.0, 5.0]
end
