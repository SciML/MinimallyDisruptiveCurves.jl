"""
    AbstractCost

Abstract interface for scalar cost functions used by minimally disruptive
curves.

To define a custom cost, subtype `AbstractCost` and extend the two public
generic functions `value` and `gradient!` in this module. The default
`value_and_gradient!` implementation composes those methods.

# Required Methods
- `value(cost, z)::Number`: return the scalar cost at physical parameters `z`.
- `gradient!(cost, g, z)`: overwrite `g` with the gradient with respect to
  `z`, then return `g`.

`z` must be accepted as an `AbstractVector`-like parameter container. `g`
must have the same length as `z`; implementations may mutate only `g` and
their own internal caches. Do not mutate `z`. Implement
`value_and_gradient!` only when evaluating value and gradient together can
reuse work.

# Example
```julia
struct SquaredDistance <: AbstractCost
    center::Vector{Float64}
end
MinimallyDisruptiveCurves.value(cost::SquaredDistance, z) = sum(abs2, z .- cost.center) / 2
function MinimallyDisruptiveCurves.gradient!(cost::SquaredDistance, g, z)
    @. g = z - cost.center
    return g
end
```
"""
abstract type AbstractCost end

# ====================================================================
# --- Core Cost Function ---
# ====================================================================


"""
    CostFunction(f, grad!)
    CostFunction(f, grad!, fg)
    CostFunction(fg)

Wrap a user-supplied cost function (and its gradient) into a callable struct
for use inside a `TransformedCost` and ultimately an `MDCProblem`.

# Fields
- `f`: value function with signature `f(θ)::Number`.
- `grad!`: gradient function with signature `grad!(g, θ)` that overwrites and
  returns `g`.
- `fg`: optional combined function with signature `fg(g, θ)::Number`. It is
  `nothing` when no combined implementation is supplied.

# Constructors

## `CostFunction(f, grad!)` — separate value and gradient

- `f(θ)::Number` returns the scalar cost at `θ`.
- `grad!(g, θ)` writes the gradient of `f` at `θ` into the buffer `g` and
  returns `g`.

This is the base form. The solver hot loop calls `f` and `grad!` separately
per RHS evaluation. Fine when the cost and gradient share no expensive
forward computation (e.g. analytic costs).

## `CostFunction(f, grad!, fg)` — separate plus combined

- `fg(g, θ)::Number` writes the gradient at `θ` into `g` and returns the
  scalar cost in a single call.

When `fg` is supplied, the solver hot loop uses `value_and_gradient!`
(see Methods below), which calls `fg` once per RHS evaluation instead of
calling `f` and `grad!` separately. Use this when the gradient reuses
forward work from the cost — most importantly for simulation-based costs
where the forward solve is the expensive part and the gradient (via
adjoints or forward sensitivities) reuses it.

## `CostFunction(fg)` — convenience constructor

Supply only the combined `fg`; `f` and `grad!` are derived from it. The
derived `f` allocates a temporary gradient buffer on each call, which is
acceptable for one-off value queries outside the solver hot loop. For
allocation-sensitive standalone `value` calls, use the 3-arg form with an
explicit `f`.

# Methods

- `value(c, θ)` returns `c.f(θ)`.
- `gradient!(c, g, θ)` calls `c.grad!(g, θ)` and returns `g`.
- `value_and_gradient!(c, g, θ)` computes both at once:
    - if `c.fg === nothing`, falls back to `gradient!` then `value` (two
      user-side calls);
    - otherwise calls `c.fg(g, θ)` once (one user-side call).

`TransformedCost` uses `value_and_gradient!` internally on its 3-arg and
4-arg callable forms, so supplying `fg` halves the user-side call count per
RHS evaluation when forward computation can be shared. The 4-arg form
(`(tc::TransformedCost)(θ, gθ, gz, buffers)`) is the allocation-free hot
path used by `vectorfield(sys)`.

# In-place gradient contract

Both `grad!(g, θ)` and `fg(g, θ)` must write into the supplied buffer `g`
rather than allocating and returning a fresh array. This matches the
solver's preallocation pattern and is what keeps the 4-arg `TransformedCost`
hot path allocation-free. Python-side cost functions supplied via the
Python wrapper are adapted to this contract internally by the `PyCost`
adapter; users writing Julia directly must respect it.

# Examples

Separate value and gradient (simple analytic cost):

```julia
center = [1.0, 2.0, 3.0]
f(θ) = 0.5 * sum(abs2, θ .- center)
grad!(g, θ) = (@. g = θ - center; g)
cost = CostFunction(f, grad!)
```
Combined value and gradient (recommended for simulation-based costs where the gradient reuses the forward solve):

```julia
function fg(g, θ)
    sol = solve(ODEProblem(dynamics!, u0, tspan, θ), Tsit5())
    positions = [s[1] for s in sol.u]
    cost = sum(abs2, positions .- target_positions) / length(positions)
    # Compute gradient via adjoints or forward sensitivities, reusing `sol`.
    # ...
    @. g = computed_gradient
    return cost
end
cost = CostFunction(fg)
```
"""
struct CostFunction{F, G, FG} <: AbstractCost
    f::F
    grad!::G
    fg::FG  # may be Nothing
end

CostFunction(f, g) = CostFunction(f, g, nothing)

function CostFunction(fg)
    f = θ -> begin
        g_buf = similar(θ)
        return fg(g_buf, θ)
    end
    g! = (g, θ) -> begin
        fg(g, θ)
        return nothing
    end
    return CostFunction(f, g!, fg)
end

# Explicit 3-arg form: user supplies all three
# (no extra constructor needed — the default struct constructor handles this)


"""
    value(cost, z) -> Number

Evaluate the scalar value of an [`AbstractCost`](@ref) at physical parameters
`z`.

# Arguments
- `cost::AbstractCost`: cost implementation.
- `z`: physical-coordinate parameter vector. It is never mutated.

Custom `AbstractCost` implementations must extend this generic together with
[`gradient!`](@ref). For an allocation-conscious combined evaluation, call
[`value_and_gradient!`](@ref) instead.

# Example
```julia
cost = CostFunction(z -> sum(abs2, z), (g, z) -> (g .= 2 .* z))
value(cost, [1.0, 2.0])
```
"""
value(c::CostFunction, θ) = c.f(θ)

"""
    gradient!(c, g, θ)

Overwrite `g` with the gradient of `cost` at physical parameters `z` and
return `g`.

# Arguments
- `cost::AbstractCost`: cost implementation.
- `g`: preallocated output buffer with one entry per entry of `z`.
- `z`: physical-coordinate parameter vector. It is not mutated.

Custom `AbstractCost` implementations must extend this generic and return the
same `g` object after overwriting it.

# Example
```julia
cost = CostFunction(z -> sum(abs2, z), (g, z) -> (g .= 2 .* z))
gradient!(cost, zeros(2), [1.0, 2.0])
```
"""
gradient!(c::CostFunction, g, θ) = c.grad!(g, θ)

# Generic fallback for any AbstractCost (user-defined costs that aren't CostFunction)
"""
    value_and_gradient!(c, g, θ)

Evaluate `cost` and overwrite `g` with its gradient at `z`.

# Arguments
- `cost::AbstractCost`: cost implementation.
- `g`: preallocated gradient output buffer.
- `z`: physical-coordinate parameter vector. It is not mutated.

# Returns
The scalar cost value. The generic fallback calls [`gradient!`](@ref) and then
[`value`](@ref); custom cost types may extend this function to share work.

# Example
```julia
cost = CostFunction(z -> sum(abs2, z), (g, z) -> (g .= 2 .* z))
gradient = zeros(2)
value_and_gradient!(cost, gradient, [1.0, 2.0])
```
"""
function value_and_gradient!(c::AbstractCost, g, z)
    gradient!(c, g, z)
    return value(c, z)
end

# CostFunction with no combined fg: separate calls (same as the fallback, but specialized
# to ensure dispatch doesn't accidentally land on the parametric method below for the
# Nothing case)
function value_and_gradient!(c::CostFunction{F, G, Nothing}, g, z) where {F, G}
    gradient!(c, g, z)
    return value(c, z)
end

# CostFunction with combined fg: one user-side call
function value_and_gradient!(c::CostFunction{F, G, FG}, g, z) where {F, G, FG}
    return c.fg(g, z)
end


"""
    TransformedCost(cost, chain)
    TransformedCost(cost::CostFunction)

Associate an [`AbstractCost`](@ref) with a [`TransformChain`](@ref), allowing
an MDC to evolve in transformed coordinates while the cost is evaluated in
physical coordinates.

# Fields
- `cost::AbstractCost`: the physical-coordinate cost.
- `chain::TransformChain`: map from MDC coordinates to the cost coordinates.

# Arguments
- `cost`: an `AbstractCost` implementation.
- `chain`: transform sequence applied in declaration order. The one-argument
  constructor uses the identity `TransformChain()`.

Calling `transformed(θ)` returns the cost. Calling
`transformed(θ, gθ)` also returns the cost and overwrites `gθ` with the
gradient in transformed coordinates.

# Example
```julia
cost = CostFunction(z -> sum(abs2, z), (g, z) -> (g .= 2 .* z))
transformed = TransformedCost(cost, TransformChain(ScaleTransform([2.0, 1.0])))
transformed([1.0, 2.0])
```
"""
struct TransformedCost{C <: AbstractCost, T <: TransformChain} <: AbstractCost
    cost::C
    chain::T
end

# Value-only evaluation
(tc::TransformedCost)(θ) = value(tc.cost, forward(tc.chain, θ))

function (tc::TransformedCost)(θ, gθ, gz)
    z = forward(tc.chain, θ)
    c_val = value_and_gradient!(tc.cost, gz, z)
    g_transformed = pullback!(tc.chain, gz, z)
    gθ .= g_transformed
    return c_val
end

# Fallback ONLY for users calling it outside the solver loop
function (tc::TransformedCost)(θ, gθ)
    z = forward(tc.chain, θ)
    gz = similar(z) # Acceptable for one-off manual calls
    return tc(θ, gθ, gz)
end

function (tc::TransformedCost)(θ, gθ, gz, buffers)
    z = forward!(tc.chain, buffers, θ)
    c_val = value_and_gradient!(tc.cost, gz, z)
    pullback!(tc.chain, gθ, gz, buffers)
    return c_val
end
