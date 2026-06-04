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

# Value and Gradient evaluation (in-place for gθ)
function (tc::TransformedCost)(θ, gθ)
    # 1. Forward pass to the cost function's native physical space
    z = forward(tc.chain, θ)
    
    # 2. Compute gradient in the physical space (gz = dL/dz)
    gz = similar(z) 
    gradient!(tc.cost, gz, z)
    
    # 3. Pull back the gradient to the optimization space (g_transformed = dL/dθ)
    # Uses the optimized forward-cached sweep from transforms.jl
    g_transformed = pullback!(tc.chain, gz, z)
    
    # 4. Copy the safely pulled-back gradient into our operational buffer
    gθ .= g_transformed
    
    return value(tc.cost, z)
end


