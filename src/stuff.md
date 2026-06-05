abstract type AbstractTransform end

# ====================================================================
# --- Transform Chain ---
# ====================================================================
struct TransformChain{T<:Tuple} <: AbstractTransform
    ts::T
end
TransformChain(ts::AbstractTransform...) = TransformChain(ts)

function forward(tc::TransformChain, x)
    for t in tc.ts
        x = forward(t, x)
    end
    return x
end

function inverse(tc::TransformChain, y)
    for i in length(tc.ts):-1:1
        y = inverse(tc.ts[i], y)
    end
    return y
end

function pullback!(tc::TransformChain, g_initial, y_final)
    return _pullback_recursive(tc.ts, g_initial, y_final)
end

@inline function _pullback_recursive(ts::Tuple, g_out, y)
    # Split into the last element and all preceding elements
    init = Base.front(ts)
    last_t = Base.last(ts)
    
    # Compute the input parameter 'x' for the last layer
    x = inverse(last_t, y)
    
    # Compute the structural gradient coming into this layer
    g_in = similar(x)
    pullback!(last_t, g_in, g_out, x, y)
    
    # Recurse backwards through the remainder of the chain
    return _pullback_recursive(init, g_in, x)
end

# Base Case termination loop
@inline _pullback_recursive(::Tuple{}, g_out, y) = g_out




function forward(tc::TransformChain{Tuple{}}, x)
    return x
end

function inverse(tc::TransformChain{Tuple{}}, y)
    return y
end

function pullback!(tc::TransformChain{Tuple{}}, g_initial, y_final)
    return g_initial
end

function transform_names(tc::TransformChain{Tuple{}}, names::Vector{Symbol})
    return names
end

# ====================================================================
# --- Primitive Transforms ---
# ====================================================================

# --- Scaling ---
struct ScaleTransform{V<:AbstractVector{Float64}} <: AbstractTransform
    w::V
end
forward(t::ScaleTransform, x) = x .* t.w
inverse(t::ScaleTransform, y) = y ./ t.w

function pullback!(t::ScaleTransform, g_in, g_out, x, y)
    @. g_in = g_out * t.w
    return g_in
end

# --- LogAbs ---
struct LogAbsTransform <: AbstractTransform end
forward(::LogAbsTransform, x) = log.(abs.(x))
inverse(::LogAbsTransform, y) = exp.(y)

function pullback!(::LogAbsTransform, g_in, g_out, x, y)
    @. g_in = g_out / x
    return g_in
end

# --- Free/Fixed Parameter Masking ---
struct FixedParamsTransform <: AbstractTransform
    free_idx::Vector{Int}
    fixed_idx::Vector{Int}
    fixed_vals::Vector{Float64}
    full_dim::Int
end

function FixedParamsTransform(free_idx::Vector{Int}, fixed_vals::Vector{Float64}, full_dim::Int)
    fixed_idx = setdiff(1:full_dim, free_idx)
    if length(fixed_idx) != length(fixed_vals)
        error("Dimension mismatch: Got $(length(fixed_vals)) fixed values, but calculated $(length(fixed_idx)) fixed indices.")
    end
    return FixedParamsTransform(free_idx, fixed_idx, fixed_vals, full_dim)
end

function forward(t::FixedParamsTransform, x::AbstractVector)
    if length(x) != length(t.free_idx)
        error("Dimension Mismatch in forward: Input size ($(length(x))) must match number of free indices ($(length(t.free_idx))).")
    end
    x_full = zeros(eltype(x), t.full_dim)
    x_full[t.free_idx] .= x
    x_full[t.fixed_idx] .= t.fixed_vals
    return x_full
end

function inverse(t::FixedParamsTransform, y::AbstractVector)
    if length(y) != t.full_dim
        error("Dimension Mismatch in inverse: Input size ($(length(y))) must match full_dim ($(t.full_dim)).")
    end
    return y[t.free_idx]
end

function pullback!(t::FixedParamsTransform, g_in, g_out, x, y)
    @views g_in .= g_out[t.free_idx]
    return g_in
end

# ====================================================================
# --- Parameter Name Tracking Metadata Pipeline ---
# ====================================================================

function transform_names(t::LogAbsTransform, names::Vector{Symbol})
    return [Symbol("log(abs($(n)))") for n in names]
end

function transform_names(t::ScaleTransform, names::Vector{Symbol})
    return [t.w[i] == 1.0 ? names[i] : Symbol("$(t.w[i]) * $(names[i])") for i in 1:length(names)]
end

function transform_names(t::FixedParamsTransform, names::Vector{Symbol})
    # Dynamically adapts: Slices names vector if matched to full physical dimensions
    if length(names) == t.full_dim
        return names[t.free_idx]
    end
    return names
end

function transform_names(chain::TransformChain, names::Vector{Symbol})
    current_names = copy(names)
    for transform in chain.ts
        current_names = transform_names(transform, current_names)
    end
    return current_names
end


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

