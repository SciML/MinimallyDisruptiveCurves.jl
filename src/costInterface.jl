abstract type AbstractCost end

# ====================================================================
# --- Core Cost Function ---
# ====================================================================
"""
    CostFunction(cost, grad!)
- `cost(θ)` should return a scalar value
- `grad!(g, θ)` should write the gradient of `cost(θ)` to `g`
This gets wrapped in `TransformedCost` when uesd in an MDCurve, to allow for chained transforms of the parameter space
"""
struct CostFunction{F, G} <: AbstractCost
    f::F
    grad!::G
end

value(c::CostFunction, θ) = c.f(θ)
gradient!(c::CostFunction, g, θ) = c.grad!(g, θ)

"""
    TransformedCost(cost, chain)
Wraps a `CostFunction` type. Applies the chain of transforms, each subtypes of `AbstractTransform` to the cost function, to alter the co-ordinate system the MD curve traces through
"""
struct TransformedCost{C <: AbstractCost, T <: TransformChain} <: AbstractCost
    cost::C
    chain::T
end

# Value-only evaluation
(tc::TransformedCost)(θ) = value(tc.cost, forward(tc.chain, θ))

# caches intermediates in gz
function (tc::TransformedCost)(θ, gθ, gz)
    z = forward(tc.chain, θ)
    gradient!(tc.cost, gz, z)
    g_transformed = pullback!(tc.chain, gz, z)
    gθ .= g_transformed
    return value(tc.cost, z)
end

# Fallback ONLY for users calling it outside the solver loop
function (tc::TransformedCost)(θ, gθ)
    z = forward(tc.chain, θ)
    gz = similar(z) # Acceptable for one-off manual calls
    return tc(θ, gθ, gz)
end
