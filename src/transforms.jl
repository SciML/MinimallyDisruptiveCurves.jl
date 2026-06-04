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
    N_layers = length(tc.ts)
    if N_layers == 0
        return g_initial
    end

    # Allocated locally per call -> 100% thread-isolated and safe!
    xs = Vector{typeof(y_final)}(undef, N_layers)
    ys = Vector{Any}(undef, N_layers)
    
    # 1. Trace the intermediate states from top to bottom
    current = y_final
    for i in N_layers:-1:1
        ys[i] = current
        xs[i] = inverse(tc.ts[i], current)
        current = xs[i]
    end
    
    # 2. Reverse sweep to compute compound sensitivities
    cur_g_out = g_initial
    for i in N_layers:-1:1
        t = tc.ts[i]
        g_in = similar(xs[i]) # Clean local allocation
        pullback!(t, g_in, cur_g_out, xs[i], ys[i])
        cur_g_out = g_in
    end
    
    return cur_g_out
end




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


