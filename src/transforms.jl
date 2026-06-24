"""
   Abstract type for transforms that can be chained and applied to a user-supplied cost function inside a TransformChain 
"""
abstract type AbstractTransform end

"""
    
Accepts a tuple of transforms. Chains them together to make a composite transform.
"""
struct TransformChain{T <: Tuple} <: AbstractTransform
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

    # Compute the gradient coming into this layer
    g_in = similar(x)
    pullback!(last_t, g_in, g_out, x, y)

    # Recurse backwards through the remainder of the chain
    return _pullback_recursive(init, g_in, x)
end

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
# --- Bais Transforms ---
# ====================================================================

"""
   Scales parameters by constants. Correspondingly scales the effort MD curves take to move that parameter (which is the point of this)
"""
struct ScaleTransform{V <: AbstractVector{<:Real}} <: AbstractTransform
    w::V
end
forward(t::ScaleTransform, x) = x .* t.w
inverse(t::ScaleTransform, y) = y ./ t.w

function pullback!(t::ScaleTransform, g_in, g_out, x, y)
    @. g_in = g_out * t.w
    return g_in
end

"""
Transforms the parameters by x -> log(abs(x)) (componentwise).
- This means that MDCs trace through relative, rather than absolute, changes in parameters.
- Means that parameter cannot cross zero (in their raw co-ordinates)
"""
struct LogAbsTransform <: AbstractTransform end

# Forward: Optimizer Space (log) -> Physical Space (exp)
forward(::LogAbsTransform, x) = exp.(x)

# Inverse: Physical Space -> Optimizer Space
inverse(::LogAbsTransform, y) = log.(abs.(y))

# Pullback: z = exp(x), so dz/dx = exp(x) = z.
# g_in = g_out * z
function pullback!(::LogAbsTransform, g_in, g_out, x, y)
    @. g_in = g_out * y  # 'y' is the output of forward (the physical values)
    return g_in
end

"""
Fixes parameters that the user doesn't wish to change over the MD curve.
"""
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
# --- Name Tracking Metadata Pipeline ---
# ====================================================================

function transform_names(t::LogAbsTransform, names::Vector{Symbol})
    return [Symbol("log(abs($(n)))") for n in names]
end

function transform_names(t::ScaleTransform, names::Vector{Symbol})
    return [t.w[i] ≈ 1.0 ? names[i] : Symbol("$(t.w[i]) * $(names[i])") for i in 1:length(names)]
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


# ====================================================================
# --- Add in-place forward methods for each transform ---
# ====================================================================
function forward!(out, t::ScaleTransform, x)
    @. out = x * t.w
    return out
end

function forward!(out, ::LogAbsTransform, x)
    @. out = exp(x)
    return out
end

function forward!(out, t::FixedParamsTransform, x)
    fill!(out, zero(eltype(out)))
    @views out[t.free_idx] .= x
    @views out[t.fixed_idx] .= t.fixed_vals
    return out
end

# In-place forward for the chain (recursive for type stability)
function forward!(chain::TransformChain, buffers::Tuple, x)
    return _forward_chain!(chain.ts, buffers, x)
end

@inline _forward_chain!(::Tuple{}, ::Tuple{}, x) = x
@inline function _forward_chain!(ts::Tuple, buffers::Tuple, x)
    out = first(buffers)
    forward!(out, first(ts), x)
    return _forward_chain!(Base.tail(ts), Base.tail(buffers), out)
end

# In-place pullback for the chain (recursive for type stability)
function pullback!(tc::TransformChain, g_final, g_out, buffers)
    current_g = _pullback_chain!(tc.ts, g_out, buffers)
    g_final .= current_g
    return g_final
end

@inline _pullback_chain!(::Tuple{}, g_out, ::Tuple{}) = g_out
@inline function _pullback_chain!(ts::Tuple, g_out, buffers::Tuple)
    t = last(ts)
    y = last(buffers)
    init_ts = Base.front(ts)
    init_buffers = Base.front(buffers)

    g_in = y # Reuse y as buffer for g_in
    pullback!(t, g_in, g_out, y, y)

    return _pullback_chain!(init_ts, g_in, init_buffers)
end
