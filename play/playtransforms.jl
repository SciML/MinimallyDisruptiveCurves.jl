############################
# 1. COST FUNCTION
############################

# raw cost: C(z) = sum(z^2)
cost(z) = sum(abs2, z)

# gradient in-place: ∇C = 2z
function cost_grad!(g, z)
    @. g = 2z
    return sum(abs2, z)
end

Craw = CostFunction(cost, cost_grad!)


############################
# 2. TEST SETUP
############################

θ = [1.2, -0.7, 2.5]

chain = TransformChain((
    LogAbsTransform(),
    ScaleTransform([2.0, 0.5, 1.0])
))

tc = TransformedCost(Craw, chain)

g = similar(θ)

############################
# 3. RUN TEST
############################
θ_new = inverse(chain, θ)
println("θ_new = ", θ_new)

CC = tc(θ_new, g)

println("cost = ", CC)
println("gradient = ", g)




