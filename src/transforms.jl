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

# Non-allocating or pre-buffered pullback is ideal, but if keeping it simple for now:
function pullback!(tc::TransformChain, g_initial, y)
    cur_y = y
    cur_g_out = g_initial
    
    for i in length(tc.ts):-1:1
        t = tc.ts[i]
        
        # Reconstruct the input to this layer
        x = inverse(t, cur_y)
        
        # Allocate mutable buffer (Note: Consider passing these in via a workspace later!)
        g_in = similar(x)
        pullback!(t, g_in, cur_g_out, x, cur_y)
        
        # Step backward
        cur_g_out = g_in
        cur_y = x
    end
    
    return cur_g_out
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
    return FixedParamsTransform(free_idx, fixed_idx, fixed_vals, full_dim)
end

# Forward: Reduced space (free) -> Full space (padded)
function forward(t::FixedParamsTransform, x)
    x_full = zeros(eltype(x), t.full_dim)
    x_full[t.free_idx] .= x
    x_full[t.fixed_idx] .= t.fixed_vals
    return x_full
end

# Inverse: Full space -> Reduced space
inverse(t::FixedParamsTransform, y) = y[t.free_idx]

# Pullback: Full space gradient (g_out) -> Reduced space gradient (g_in)
function pullback!(t::FixedParamsTransform, g_in, g_out, x, y)
    @views g_in .= g_out[t.free_idx]
    return g_in
end
