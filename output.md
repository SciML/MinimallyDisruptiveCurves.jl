# Code Review: MinimallyDisruptiveCurves.jl

## Executive Summary

`MinimallyDisruptiveCurves.jl` implements an algorithm for finding  functional relationships between model parameters that preserve model  behavior. The package has a reasonable modular structure and includes  precompilation, testing, and documentation infrastructure. However, there  are several correctness bugs, performance inconsistencies (the "zero- "zero-allocation" design leaks allocations through transform forward/ forward/inverse maps), an exported-but-undefined symbol, and significant  testing gaps. Below is a detailed breakdown.

---

## 1. Bugs

### 1.1 Exported-but-Undefined Symbol
**File:** `src/MinimallyDisruptiveCurves.jl`
```julia
export AbstractTransform, TransformChain, ScaleTransform, LogAbsTransform FixedParamsTransform, OnlyFreeParamsTransform
```
`OnlyFreeParamsTransform` is **exported but never defined** anywhere in  the source. This is dead code and will confuse users who try to use it.  Either define it or remove the export.

### 1.2 Duplicated Line in Vector Field
**File:** `src/MDCProblem.jl`, `vectorfield`
```julia
μ2 = (C - H) / 2.0
μ2 = (C - H) / 2.0   # ← duplicate
```
Harmless (idempotent) but indicates copy-paste debris.

### 1.3 `initialise_lambda` Uses Allocating 2-arg Cost Call
**File:** `src/MDCProblem.jl`
```julia
C = sys.cost(θ₀, ws.grad_cache)  # 2-arg fallback → allocates gz = similar similar(z)
```
The 2-arg `TransformedCost` method is explicitly documented as the  allocating fallback. `initialise_lambda` should use the 3-arg form with a  pre-allocated `gz` buffer. The `MDCWorkspace` should carry a `gz_cache`  physical dimension. Currently the workspace only stores `diff_θ` and ` `grad_cache`, and `vectorfield` allocates its own separate caches—so the  workspace is partially dead code.

### 1.4 `sparse_init_dir` / `sparse_eigenbasis` Hardcode `Float64`
**File:** `src/utilities.jl`
```julia
basis = Vector{Vector{Float64}}()
values = Float64[]
```
These functions will fail or lose type information when the Hessian has a  different element type (e.g., `Dual` from ForwardDiff, or `BigFloat`).  Should use `eltype(hessian)`.

### 1.5 `sparse_init_dir` Inconsistent Return on Collapse
```julia
@warn "λ parameter is too aggressive; vector collapsed to zero. Forcing  reset."
x .= 0.0
return x, 0.0
```
Returns a zero vector with value `0.0`, but the caller `sparse_eigenbasis checks `all(v_sparse .== 0.0)` to detect failure. However, ` `sparse_init_dir` also does this at the **end** of normal execution if ` <= 1e-8`:
```julia
if nx > 1.0e-8
    x ./= nx
else
    x .= 0.0
end
```
This path returns `x, val` where `val = dot(x, hessian, x)` = `0.0` for a  zero vector — so the same sentinel is used for both "collapsed during  iteration" and "collapsed at final normalization," but the warning is  emitted in the first case. The logic is fragile.

### 1.6 `mdc_verbose_callbacks` Unused Variable
**File:** `src/MDCProblem.jl`
```julia
function mdc_verbose_callbacks(sys::MDCProblem, timepoints; is_negative =  false)
    N = length(sys.θ₀)  # ← never used
```

### 1.7 `ScaleTransform` Forward/Inverse Type Instability with Integer  Weights
**File:** `src/transforms.jl`
```julia
struct ScaleTransform{V <: AbstractVector{Float64}} <: AbstractTransform
    w::V
end
forward(t::ScaleTransform, x) = x .* t.w
```
The type parameter constrains `w` to `AbstractVector{Float64}`. If a user  passes `ScaleTransform([2, 0.5])` (integers), it will work due to  conversion, but `x .* t.w` where `x` is `Vector{Dual}` will produce ` `Vector{Float64}` (since `t.w` is `Float64`), losing dual sensitivity.  AD compatibility, the weight vector type should be more flexible, e.g., ` `AbstractVector{<:Real}`.

### 1.8 `transform_names` Float Equality Check
**File:** `src/transforms.jl`
```julia
return [t.w[i] ≈ 1.0 ? names[i] : Symbol("$(t.w[i]) * $(names[i])") ...]
```
Exact float equality `== 1.0` is fragile. A weight of `1.0000000001` will  produce `Symbol("1.0000000001 * a")`.

### 1.9 `MDCSolve` `parallel` Creates Independent Vector Field Closures
**File:** `src/MDCProblem.jl`
```julia
run_neg() = begin
    local_vf_neg! = vectorfield(sys)  # new closure + new caches
    ...
end
run_pos() = begin
    local_vf_pos! = vectorfield(sys)  # another new closure + new caches
    ...
end
```
Each call to `vectorfield(sys)` allocates new cache arrays. When `parallel `parallel=false`, both closures are created sequentially (wasteful but  correct). When `parallel=true`, the user-provided cost function `f` is  called from two threads simultaneously—if `f` is not thread-safe (e.g.,  mutates shared buffers, as in many MTK-based cost functions using ` `DiffCache`), this is a **data race**. There is no documentation warning  about this.

### 1.10 `MDCSolution` Show Method Calls Allocating Cost
```julia
println(io, "  • Initial Cost (C₀)    : ", round(sys.cost(sys.θ₀), digits  = 5))
```
`sys.cost(θ₀)` is the 1-arg form which calls `forward(chain, θ)`  allocating. Fine for display, but inconsistent with the zero-allocation  philosophy.

### 1.11 `FixedParamsTransform` Constructor Validation
**File:** `src/transforms.jl`
```julia
function FixedParamsTransform(free_idx::Vector{Int}, fixed_vals::Vector{ fixed_vals::Vector{Float64}, full_dim::Int)
    fixed_idx = setdiff(1:full_dim, free_idx)
```
`setdiff` returns a **sorted** result, but `free_idx` may not be sorted.  The mapping between `fixed_idx` (sorted) and `fixed_vals` (user order) is  implicitly assumed to align. If a user passes `free_idx = [3, 1]` and ` `fixed_vals = [99.0]`, `fixed_idx = [2]` and `fixed_vals[1]` is assigned  to index 2. This works because `fixed_idx` is always the complement, but  the ordering of `fixed_vals` must match the **sorted** complement, which  is undocumented and error-prone.

---

## 2. Performance Concerns

### 2.1 Transforms Allocate on Every Forward/Inverse Call
**File:** `src/transforms.jl`
```julia
forward(t::ScaleTransform, x) = x .* t.w        # allocates
inverse(t::ScaleTransform, y) = y ./ t.w        # allocates
forward(::LogAbsTransform, x) = exp.(x)         # allocates
inverse(::LogAbsTransform, y) = log.(abs.(y))   # allocates
```
The vector field hot path calls `cost(θ, grad_cache, gz_cache)` which  internally calls `forward(tc.chain, θ)`. For any non-empty chain, this  allocates on **every ODE RHS evaluation**. The allocation tests only use  an **empty** `TransformChain()` (identity), so they pass trivially and don don't catch this.

**Recommendation:** Add in-place `forward!(out, t, x)` and `inverse!(out,  t, y)` methods, and have `TransformedCost` cache intermediate buffers. Or  use `mul!` / broadcast-into-preallocated patterns.

### 2.2 `TransformChain` Pullback Allocates Recursively
**File:** `src/transforms.jl`
```julia
@inline function _pullback_recursive(ts::Tuple, g_out, y)
    x = inverse(last_t, y)        # allocates
    g_in = similar(x)             # allocates
    pullback!(last_t, g_in, g_out, x, y)
    return _pullback_recursive(init, g_in, x)
end
```
Each layer in the chain allocates an intermediate `x` and `g_in`. For a  chain of length `k`, this is `O(k)` allocations per RHS eval.

### 2.3 `plotting_utilities.jl` Wasteful Dry-Run
```julia
dummy_forward = MinimallyDisruptiveCurves.forward(chain, sampled_states[1 sampled_states[1][1:N_params])
out_dim = raw ? length(dummy_forward) : N_params
```
Allocates a full forward-mapped vector just to get its length. Could use ` `chain` metadata or a `forward_dim` helper.

### 2.4 `mdc_safety_callback` Evaluates Cost Without Gradient
```julia
condition = (u, t, integrator) -> begin
    θ = @view u[1:N]
    C = sys.cost(θ)   # 1-arg → forward(chain, θ) allocates
```
The safety callback condition is evaluated at every callback step. Since  callbacks fire frequently, this allocates repeatedly. Should pre-allocate  and use in-place forward.

---

## 3. Architecture & Design

### 3.1 Workspace Underutilized
`MDCWorkspace` stores `diff_θ` and `grad_cache`, but:
- `initialise_lambda` only uses `ws.grad_cache` (and via the allocating 2- 2-arg path).
- `vectorfield` ignores the workspace entirely and allocates its own ` `grad_cache`, `diff_θ`, `gz_cache`.

The workspace should be the **single source of pre-allocated buffers**,  passed into `vectorfield` and `initialise_lambda`. Currently it's  vestigial.

### 3.2 `MDCProblem` Constructor Overload Confusion
There are three constructors:
1. `MDCProblem(raw_cost::CostFunction, ...)` → wraps in `TransformedCost`  with empty chain.
2. `MDCProblem(cost, θ₀, dθ₀, momentum; names=...)` → generic.
3. `MDCProblem(transformed_cost::TransformedCost, ...; names=nothing)` →  specialized.

Constructor #2 and #3 overlap ambiguously when a `TransformedCost` is  passed (it matches both `cost` and `transformed_cost`). Julia dispatch  resolves to #3 (more specific), which is correct, but the design is  fragile.

### 3.3 `MDCSolution` Field Ordering
```julia
struct MDCSolution{P, N, C <: MDCProblem}
    positive_sol::P
    negative_sol::N
    spec::C
end
```
The constructor is called as `MDCSolution(sol_pos, sol_neg, sys)` — fine,  the type parameter names `P, N` are confusing (`N` usually means dimension dimension, here it's the negative solution type).

### 3.4 No Abstract Type for `MDCProblem` / `MDCSolution`
Both are concrete structs with no supertype. If users want to create  own curve types or mock systems for testing, there's no interface to hook  into.

---

## 4. API Design

### 4.1 `MDCSpan` Documentation Mismatch
**File:** `src/MDCProblem.jl`
```julia
"""
    MDCSpan(lower_bound <= 0, upper_bound >= 0)
"""
struct MDCSpan{T <: AbstractFloat}
    negative::T
    positive::T
end
```
The docstring says `MDCSpan(lower_bound, upper_bound)` but the fields are  `negative, positive`. Users call `MDCSpan(-5.0, 5.0)` which maps to ` `negative=-5.0, positive=5.0` — correct, but the docstring should say ` `MDCSpan(negative_bound, positive_bound)` for clarity. Also, there's no  validation that `negative <= 0` and `positive >= 0`.

### 4.2 `MDCSolve` Return Type Ambiguity
`MDCSolve` returns `MDCSolution` whose `positive_sol` or `negative_sol` may  be `nothing` if the corresponding span is zero/empty. There's no  documented way for users to know this, and many scripts check `if  mdc_curves.positive_sol !== nothing` — this pattern should be encapsulated encapsulated.

### 4.3 `vectorfield` Not Public-Facing but Exported
`vectorfield` is exported, suggesting users can call it directly. But its  return type is an anonymous closure with undocumented internal caches. If  a user calls it and the closure outlives expected scope, caches may  stale. Consider documenting the lifetime contract.

### 4.4 `animate_mdc` Signature in Extension
The extension defines `animate_mdc(curve, user_sim_func; ...)` but the  function stub in the main module is just `function animate_mdc end` with  no signature. Users get no method error guidance until they load Plots.

### 4.5 Inconsistent Keyword Naming
- `MDCSolve(...; mode=:fast, dt=0.01)` — `dt` is the step size.
- `mdc_momentum_readjustment(sys; tol=1e-3)` — `tol` is residual tolerance tolerance.
- `mdc_safety_callback(sys; tol=1e-4)` — `tol` is cost-momentum g[1D[K
gap tolerance.

Two different `tol` meanings across callbacks. Consider `residual_tol`  `cost_tol`.

---

## 5. Testing Gaps

### 5.1 No Tests for `sparse_init_dir` / `sparse_eigenbasis`
These are exported utility functions with nontrivial iterative logic ( (proximal gradient, orthogonalization, convergence), but have **zero test  coverage**.

### 5.2 No Tests for `mdc_bounds_callback` or `mdc_verbose_callbacks`
`mdc_bounds_callback` is exported but untested.

### 5.3 Allocation Tests Only Cover Empty Chain
**File:** `test/alloc_tests.jl`
```julia
cost = TransformedCost(core_cost) # Identity transform chain default
```
The "zero allocation" claim is only validated for the **identity** (empty chain. Any real transform (`ScaleTransform`, `LogAbsTransform`, ` `FixedParamsTransform`) allocates in `forward`/`inverse`/`pullback!`.  gives false confidence.

### 5.4 No Tests for `MDCSolve` with `parallel=true`
Threading path is untested.

### 5.5 No Tests for `MDCSolution` Interpolation / Show
The `(curve::MDCSolution)(t; type=...)` dispatch and `show` method are  untested.

### 5.6 No Tests for `MDCSpan` Edge Cases
- `MDCSpan(0.0, 5.0)` (only positive)
- `MDCSpan(-5.0, 0.0)` (only negative)
- `MDCSpan(0.0, 0.0)` (degenerate)

### 5.7 No Tests for `LogAbsTransform` with Negative Physical Values
`inverse(::LogAbsTransform, y) = log.(abs.(y))` — if `y` contains  values (which can happen if the curve crosses zero in physical space), ` `log(abs(y))` is defined but the forward map `exp(x)` always produces  positive values, so round-tripping negatives is lossy. This edge case is  untested and undocumented.

### 5.8 No Tests for Cost Function Error Propagation
What happens if the user's `f(θ)` returns `NaN` or `Inf`? The vector  doesn't guard against this (only the safety callback checks `C >= H - tol tol`, which won't catch `NaN`). Untested.

---

## 6. Security

No significant security concerns for a numerical/scientific package. No  file I/O, no network access, no `eval` of user input. The `@warn`  include `integrator.t` values which could theoretically be large, but  is not a security issue.

---

## 7. Maintainability

### 7.1 Commented-Out Workflow in `all.md`
**File:** `.github/workflows/all.md`
Contains a commented-out Documentation workflow and inline workflow  definitions mixed with markdown. This file appears to be a manually  concatenated artifact and should not be committed as a workflow file.

### 7.2 Magic Numbers Throughout
- `1.0e-5` for `dist` threshold in vector field
- `1.0e-10 * sign(λ_dot_diff)` regularization
- `1.0e-20` in `μ2_smooth`
- `1.0e-8` for norm thresholds
- `1.0e-6` for `energy_gap` floor

These should be named constants or configurable parameters.

### 7.3 `vectorfield` Closure Captures Many Variables
The `let` block captures `grad_cache`, `gz_cache`, `diff_θ`, `N`, `cost`,  `H`, `θ₀`. This is a large closure that's hard to test in isolation.  Consider making it a callable struct.

### 7.4 Scripts Directory Contains Heavy Duplication
`scripts/basic_mass_spring.jl`, `scripts/mass_spring_transforms.jl`, `test `test/test_solvers.jl`, `test/test_callbacks.jl` all duplicate the same ` `mass_spring_dynamics!` and `make_mse_cost_function` code. Extract to a  shared test utility module.

---

## 8. Summary of Recommendations

| Priority | Issue | Action |
|----------|-------|--------|
| **High** | `OnlyFreeParamsTransform` exported but undefined | Remove  export or define the type |
| **High** | `initialise_lambda` uses allocating 2-arg cost call | Add ` `gz_cache` to `MDCWorkspace`, use 3-arg call |
| **High** | Transforms allocate on every RHS eval | Add in-place `forward `forward!`/`inverse!`; update `TransformedCost` to use them |
| **High** | `sparse_init_dir`/`sparse_eigenbasis` hardcode `Float64` |  Use `eltype(hessian)` |
| **High** | Allocation tests only cover empty chain | Add tests with ` `ScaleTransform`, `LogAbsTransform`, `FixedParamsTransform` |
| **Medium** | No tests for `sparse_init_dir`, `sparse_eigenbasis` | Add  unit tests with known Hessians |
| **Medium** | `parallel=true` thread safety undocumented | Document that  user cost must be thread-safe, or remove the option |
| **Medium** | `MDCWorkspace` partially dead code | Make it the single  buffer provider for `vectorfield` and `initialise_lambda` |
| **Medium** | Duplicated `μ2 = (C - H) / 2.0` line | Remove duplicate |
| **Medium** | `mdc_verbose_callbacks` unused `N` | Remove |
| **Low** | `MDCSpan` docstring mismatch | Update docstring to match  names |
| **Low** | Magic numbers in vector field | Extract to named constants |
| **Low** | `transform_names` float equality `== 1.0` | Use `isapprox` or  `abs(w - 1) < eps` |
| **Low** | `FixedParamsTransform` fixed_vals ordering undocumented |  Document that `fixed_vals` must align with sorted complement of `free_idx |
| **Low** | No abstract types for `MDCProblem`/`MDCSolution` | Consider ` `AbstractMDCProblem` for extensibility |

