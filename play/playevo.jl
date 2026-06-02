using OrdinaryDiffEq

# 1. Setup a dummy user-defined Cost mapping
raw_f(θ) = (θ[1] - 1.0)^2 + (θ[2] - 2.0)^2
function raw_g!(g, θ)
    g[1] = 2.0 * (θ[1] - 1.0)
    g[2] = 2.0 * (θ[2] - 2.0)
end

user_cost = CostFunction(raw_f, raw_g!)
chain = TransformChain(ScaleTransform([1.0, 1.0])) # Simple identity scale

# Wrap using your TransformedCost implementation
θ₀ = [0.0, 0.0]
transformed_cost = TransformedCost(user_cost, chain)

# 2. Package into the MDC problem definition
momentum_H = 10.0 # Must be strictly higher than your initial cost f(0,0) = 5.0
sys = MDCSystem(transformed_cost, θ₀, momentum_H)

curve = MDCsolve(sys, span=MDCSpan(-2.0, 5.0))

println("Negative Curve Completed Steps: ", length(curve.negative_sol))
println("Positive Curve Completed Steps: ", length(curve.positive_sol))
