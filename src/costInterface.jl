"""
    AbstractCost

Abstract interface for scalar cost functions used by minimally disruptive curves.
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


value(c::CostFunction, θ) = c.f(θ)

"""
    gradient!(c, g, θ)

Write the gradient of cost `c` at parameters `θ` into `g` and return `g`.
"""
gradient!(c::CostFunction, g, θ) = c.grad!(g, θ)

# Generic fallback for any AbstractCost (user-defined costs that aren't CostFunction)
"""
    value_and_gradient!(c, g, θ)

Write the gradient of cost `c` at `θ` into `g` and return the scalar cost value.
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
Wraps a `CostFunction` type. Applies the chain of transforms, each subtypes of `AbstractTransform` to the cost function, to alter the co-ordinate system the MD curve traces through
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
