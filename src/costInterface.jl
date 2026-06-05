abstract type AbstractCost end

# ====================================================================
# --- Core Cost Function ---
# ====================================================================

struct CostFunction{F, G} <: AbstractCost
    f::F
    grad!::G
end

value(c::CostFunction, θ) = c.f(θ)
gradient!(c::CostFunction, g, θ) = c.grad!(g, θ)


# ====================================================================
# --- Transformed Cost Wrapper ---
# ====================================================================

struct TransformedCost{C<:AbstractCost, T<:TransformChain} <: AbstractCost
    cost::C
    chain::T
end

# Value-only evaluation
# Maps optimization parameter θ -> physical space via forward, then evaluates
(tc::TransformedCost)(θ) = value(tc.cost, forward(tc.chain, θ))


function (tc::TransformedCost)(θ, gθ, gz)
    z = forward(tc.chain, θ)
    gradient!(tc.cost, gz, z)
    g_transformed = pullback!(tc.chain, gz, z)
    gθ .= g_transformed
    return value(tc.cost, z)
end

# Keep this fallback ONLY for users calling it outside the solver loop
function (tc::TransformedCost)(θ, gθ)
    z = forward(tc.chain, θ)
    gz = similar(z) # Acceptable for one-off manual calls
    return tc(θ, gθ, gz)
end

