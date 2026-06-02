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
(tc::TransformedCost)(θ) = value(tc.cost, forward(tc.chain, θ))

# Value and Gradient evaluation (in-place for gθ)
function (tc::TransformedCost)(θ, gθ)
    # 1. Forward pass to the cost function's native space
    z = forward(tc.chain, θ)
    
    # 2. Compute gradient in the native space (gz = dL/dz)
    gz = similar(z) 
    gradient!(tc.cost, gz, z)
    
    # 3. Pull back the gradient to the parameter space (g_transformed = dL/dθ)
    # Pass 'z' (the output) because the chain unravels backwards from the output!
    g_transformed = pullback!(tc.chain, gz, z)
    
    # 4. Copy the safely pulled-back gradient into our output buffer
    gθ .= g_transformed
    
    return value(tc.cost, z)
end


# ====================================================================
# --- Chain Extensions ---
# ====================================================================

# Computes the full inverse operation through the entire chain sequentially
function inverse(tc::TransformChain, x)
    @inbounds for i in length(tc.ts):-1:1
        x = inverse(tc.ts[i], x)
    end
    return x
end
