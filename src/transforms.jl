# Type-stable tuple helpers mirroring the (non-public) Base.front / Base.tail:
# _mdc_front drops the last element, _mdc_tail drops the first.
@inline _mdc_tail(t::Tuple) = _mdc_argtail(t...)
@inline _mdc_argtail(_, rest...) = rest

@inline _mdc_front(t::Tuple) = _mdc_front_impl(t...)
@inline _mdc_front_impl(v) = ()
@inline _mdc_front_impl(v, t...) = (v, _mdc_front_impl(t...)...)

"""
Abstract interface for coordinate transforms used by [`TransformChain`](@ref).

To define a custom transform, subtype `AbstractTransform` and extend the
public generics [`forward`](@ref), [`inverse`](@ref), and [`pullback!`](@ref).

# Required Methods
- `forward(transform, x)`: return physical coordinates `y` for transformed
  coordinates `x` without mutating `x`.
- `inverse(transform, y)`: return coordinates `x` such that
  `forward(transform, x)` is `y` on the transform's valid domain.
- `pullback!(transform, g_in, g_out, x, y)`: overwrite `g_in` with
  `J_forward(x)' * g_out`, where `y == forward(transform, x)`, and return
  `g_in`.

The input/output dimensions may differ. `g_in` must have the input dimension,
`g_out` the output dimension; implementations may mutate only `g_in` and
their own caches. A generic [`forward!`](@ref) fallback is available for
custom transforms, while specialized implementations can provide an in-place
method for performance. `g_in` is distinct from `x` and `y` when a custom
transform is evaluated through a `TransformChain`.

# Example
```julia
struct ShiftTransform <: AbstractTransform
    shift::Float64
end
MinimallyDisruptiveCurves.forward(t::ShiftTransform, x) = x .+ t.shift
MinimallyDisruptiveCurves.inverse(t::ShiftTransform, y) = y .- t.shift
function MinimallyDisruptiveCurves.pullback!(::ShiftTransform, g_in, g_out, x, y)
    return copyto!(g_in, g_out)
end
```
"""
abstract type AbstractTransform end

"""
    TransformChain(transforms...)

Compose zero or more [`AbstractTransform`](@ref)s.

# Fields
- `ts::Tuple`: transforms in forward application order. This field is public
  for inspection; treat the contained transforms as immutable configuration.

# Arguments
- `transforms::AbstractTransform...`: transforms mapping MDC coordinates to
  physical cost coordinates. `forward` applies them left-to-right; `inverse`
  and `pullback!` apply them right-to-left.

An empty chain is the identity transform.

# Example
```julia
chain = TransformChain(ScaleTransform([2.0, 0.5]), LogAbsTransform())
forward(chain, [0.0, log(4.0)])
```
"""
struct TransformChain{T <: Tuple} <: AbstractTransform
    ts::T
end
TransformChain(ts::AbstractTransform...) = TransformChain(ts)

"""
    forward(transform, x)

Map transformed coordinates `x` to physical coordinates.

# Arguments
- `transform::AbstractTransform`: transform or transform chain.
- `x`: input coordinate vector. It is not mutated.

# Returns
The mapped coordinate vector. A `TransformChain` applies its transforms in
declaration order.

Custom `AbstractTransform` implementations must extend this generic.

# Example
```julia
forward(ScaleTransform([2.0, 0.5]), [3.0, 4.0])
```
"""
function forward(tc::TransformChain, x)
    for t in tc.ts
        x = forward(t, x)
    end
    return x
end

"""
    inverse(transform, y)

Map physical coordinates `y` back to transformed coordinates.

# Arguments
- `transform::AbstractTransform`: transform or transform chain.
- `y`: output-coordinate vector. It is not mutated.

# Returns
The inverse-mapped vector. A `TransformChain` applies inverses in reverse
declaration order.

# Example
```julia
inverse(ScaleTransform([2.0, 0.5]), [6.0, 2.0])
```
"""
function inverse(tc::TransformChain, y)
    for i in length(tc.ts):-1:1
        y = inverse(tc.ts[i], y)
    end
    return y
end

"""
    pullback!(transform, g_in, g_out, x, y)

Pull a physical-coordinate gradient back through `transform`.

# Arguments
- `transform::AbstractTransform`: transform or transform chain.
- `g_in`: preallocated input-coordinate gradient buffer for the four-argument
  form.
- `g_out`: output-coordinate gradient.
- `x`: transform input and `y`: transform output for the four-argument form.

# Returns
The input-coordinate gradient. `pullback!(chain, g_out, y)` allocates
intermediate buffers for a convenient one-off calculation. The four-argument
form is the extension contract and must return the same `g_in` object.

# Example
```julia
gradient = zeros(2)
pullback!(ScaleTransform([2.0, 0.5]), gradient, [3.0, 4.0], [1.0, 2.0], [2.0, 1.0])
```
"""
function pullback!(tc::TransformChain, g_initial, y_final)
    return _pullback_recursive(tc.ts, g_initial, y_final)
end

@inline function _pullback_recursive(ts::Tuple, g_out, y)
    # Split into the last element and all preceding elements
    init = _mdc_front(ts)
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
# --- Basic Transforms ---
# ====================================================================

"""
    ScaleTransform(w)

Scale each coordinate by the corresponding entry of `w`.

# Fields
- `w::AbstractVector{<:Real}`: multiplicative scale factors. Its length must
  match the transformed vectors.

# Arguments
- `w`: scale factors mapping `x` to `x .* w`.

# Example
```julia
forward(ScaleTransform([2.0, 0.5]), [3.0, 4.0])
```
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
    LogAbsTransform()

Represent positive physical coordinates in log coordinates.

`forward(transform, x)` evaluates `exp.(x)` and `inverse(transform, y)`
evaluates `log.(abs.(y))`. The forward map is strictly positive, so this
transform cannot represent a path that crosses zero.

# Example
```julia
forward(LogAbsTransform(), [0.0, log(2.0)])
```
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
    FixedParamsTransform(free_idx, fixed_vals, full_dim)

Embed a free-coordinate vector into a physical vector while holding the
remaining coordinates fixed.

# Fields
- `free_idx`: physical indices supplied by the input vector.
- `fixed_idx`: complementary physical indices, derived by the constructor.
- `fixed_vals`: values inserted at `fixed_idx`.
- `full_dim`: physical vector length.

# Arguments
- `free_idx::Vector{Int}`: distinct free physical indices.
- `fixed_vals::Vector{Float64}`: fixed values ordered by complementary index.
- `full_dim::Int`: positive physical-coordinate dimension.

# Example
```julia
forward(FixedParamsTransform([1, 3], [10.0], 3), [2.0, 4.0])
```
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
"""
    forward!(out, transform, x)
    forward!(chain, buffers, x)

In-place form of [`forward`](@ref), using caller-provided storage.

# Arguments
- `out`: output buffer with the transform output dimension.
- `transform`: an `AbstractTransform` for the three-argument form.
- `x`: input coordinates.
- `chain`: a `TransformChain` for the three-argument chain form.
- `buffers`: one output buffer per transform in `chain`.

# Returns
The final output buffer. The generic transform fallback calls `forward` and
copies its result into `out`; custom transforms can extend it to avoid that
temporary allocation.

# Example
```julia
out = zeros(2)
forward!(out, ScaleTransform([2.0, 0.5]), [3.0, 4.0])
```
"""
function forward!(out, transform::AbstractTransform, x)
    copyto!(out, forward(transform, x))
    return out
end

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
    return _forward_chain!(_mdc_tail(ts), _mdc_tail(buffers), out)
end

# The built-in implementations do not inspect `x`, so intermediate forward
# buffers can also serve as gradient storage on the allocation-free hot path.
const _BuiltinTransform = Union{ScaleTransform, LogAbsTransform, FixedParamsTransform}

function pullback!(
        tc::TransformChain{<:Tuple{Vararg{_BuiltinTransform}}},
        g_final,
        g_out,
        buffers
    )
    return _pullback_builtin_chain!(tc.ts, g_out, buffers, g_final)
end

@inline _pullback_builtin_chain!(::Tuple{}, g_out, ::Tuple{}, g_final) =
    (g_final .= g_out; g_final)

@inline function _pullback_builtin_chain!(ts::Tuple, g_out, buffers::Tuple, g_final)
    t = last(ts)
    y = last(buffers)
    init_ts = _mdc_front(ts)
    init_buffers = _mdc_front(buffers)

    # Reuse the previous layer's output buffer to store the gradient,
    # since the input dimension of `t` matches the output dimension of the previous layer.
    # If it's the first layer, we use `g_final` directly, which has the correct initial dimension.
    g_in = isempty(init_buffers) ? g_final : last(init_buffers)

    pullback!(t, g_in, g_out, y, y)

    return _pullback_builtin_chain!(init_ts, g_in, init_buffers, g_final)
end

function pullback!(tc::TransformChain, g_final, g_out, buffers)
    return _pullback_generic_chain!(tc.ts, g_out, buffers, g_final)
end

@inline _pullback_generic_chain!(::Tuple{}, g_out, ::Tuple{}, g_final) =
    (g_final .= g_out; g_final)

@inline function _pullback_generic_chain!(ts::Tuple, g_out, buffers::Tuple, g_final)
    t = last(ts)
    y = last(buffers)
    init_ts = _mdc_front(ts)
    init_buffers = _mdc_front(buffers)
    x = isempty(init_buffers) ? inverse(t, y) : last(init_buffers)
    g_in = isempty(init_ts) ? g_final : similar(x)

    pullback!(t, g_in, g_out, x, y)

    return _pullback_generic_chain!(init_ts, g_in, init_buffers, g_final)
end
