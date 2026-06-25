# MinimallyDisruptiveCurves.jl v0.4.0

## Overview

v0.4.0 is a ground-up rewrite of the public API. Code written against v0.3.x will not work without modification, and in many cases there is no one-for-one substitution: the conceptual model has changed, not just the function names.

## Architecture

The package is organized around five composable pieces:

1. **Transform chains** (`TransformChain`) — composable reparameterizations
   of the optimization space (`ScaleTransform`, `LogAbsTransform`,
   `FixedParamsTransform`). Each transform exposes `forward`, `inverse`, and
   `pullback!` so gradients flow correctly through the chain. The MDC traces
   through the transformed (optimizer) space; the user's cost is evaluated in
   physical space.

2. **Cost interface** (`AbstractCost`, `CostFunction`, `TransformedCost`) —
   user-supplied cost and gradient are wrapped in a `CostFunction` and then
   in a `TransformedCost`, which threads the user's gradient through the
   transform chain via pullback. `CostFunction` supports an optional combined
   `fg` callable that computes value and gradient in a single call, used by
   the solver hot loop via `value_and_gradient!` — useful for simulation-based
   costs where the gradient reuses the forward solve.

3. **Problem and solver** (`MDCProblem`, `MDCSpan`,
   `MDCSolution`, `MDCSolve`) — `MDCProblem` packages the cost, initial point,
   initial direction, momentum budget, and parameter names. `MDCSolve`
   integrates the Hamiltonian vector field forwards and backwards in
   arc-length from `[θ₀; λ₀]`. The solver algorithm is configurable via the
   `alg` keyword (default `Tsit5()`); any `OrdinaryDiffEq.AbstractODEAlgorithm`
   may be supplied.

4. **Callbacks** (`mdc_safety_callback`, `mdc_momentum_readjustment`,
   `mdc_bounds_callback`, `mdc_verbose_callbacks`) — `DiscreteCallback`
   factories for the standard numerical-stability concerns: cost approaching
   the momentum budget, drift of the `dH/du = 0` identity, parameter bounds,
   and logging.

5. **Initialization utilities** (`sparse_init_dir`, `sparse_eigenbasis`) —
   soft-thresholded iterative eigenvector finders for producing sparse
   initial directions from a Hessian, with optional orthogonality
   constraints.


## Public API

### Transforms
- `TransformChain(ts...)` — composable chain
- `ScaleTransform(w)`, `LogAbsTransform()`,
  `FixedParamsTransform(free_idx, fixed_vals, full_dim)`
- `forward(chain, x)`, `inverse(chain, y)`, `pullback!(chain, g, y)`
- `forward!(chain, buffers, x)`, `generate_fwd_caches(chain, θ₀)` —
  in-place hot-path variants
- `transform_names(chain, names)` — metadata propagation for plotting

### Costs
- `CostFunction(f, grad!)` — value and gradient supplied separately
- `CostFunction(f, grad!, fg)` — separate plus combined value/gradient
- `CostFunction(fg)` — convenience: derive `f` and `grad!` from combined `fg`
- `TransformedCost(cost, chain)` — wraps a cost with a transform chain
- `value(cost, θ)`, `gradient!(cost, g, θ)`
- `value_and_gradient!(cost, g, θ)` — combined evaluation; uses `fg` if
  available, otherwise falls back to separate `gradient!` + `value`

### Problem and solve
- `MDCProblem(cost, θ₀, dθ₀, H; names=...)`
- `MDCWorkspace(sys)`
- `MDCSpan(negative, positive)`
- `MDCSolve(sys; span, mode, dt, callback, parallel, alg=Tsit5()) -> MDCSolution`
- `curve(t; type=:all|:parameters|:costates)` — continuous interpolation
  across the split-span trajectory
- `cost_trajectory(curve, ts)` — evaluate the cost along the curve at
  arc-lengths `ts`; cost recovered from `curve.spec.cost`

### Callbacks
- `mdc_safety_callback(sys; tol)` — terminate when cost approaches `H`
- `mdc_momentum_readjustment(sys; tol)` — project `λ` back onto the
  `dH/du = 0` manifold when numerical drift exceeds tolerance
- `mdc_bounds_callback(ids, lbs, ubs)` — terminate on parameter bound
  violations
- `mdc_verbose_callbacks(sys, timepoints; is_negative=false)` — logging
- `mdc_dHdu_residual(sys, u, t)` — raw L1 drift diagnostic

### Initialization
- `sparse_init_dir(hessian; orthogonal_to, λ, start, ...) -> (v, val)`
- `sparse_eigenbasis(hessian, num_vectors; λ, ...) -> (basis, values)`

### Plotting
- A `RecipesBase` recipe for `MDCSolution` with keywords `max_lines`,
  `mode` (`:absolute`/`:relative`), `raw` (transformed vs physical space),
  `density`.
- `animate_mdc(curve, user_sim_func; ...)` via the `MDCPlotsExt` extension,
  loaded when `Plots` is available. Produces a multi-panel animation:
  parameter traces, instantaneous deltas, and a user-supplied simulation
  plot.


## Migration from v0.3.x

v0.4.0 is not a drop-in replacement. The closest analogues are listed below;
where there is no analogue, the capability has been removed and the user is
expected to handle it on their side.

### Entry point

- `evolve(c, Tsit5; mdc_callback=..., callback=..., saved_values=...,
  momentum_tol=...)` → `MDCSolve(sys; span=..., mode=..., dt=...,
  callback=..., parallel=..., alg=Tsit5())`.
  The solver algorithm is now configurable via the `alg` keyword (default
  `Tsit5()`); pass any `OrdinaryDiffEq.AbstractODEAlgorithm` for stiff or
  specialized problems. The `mdc_callback` keyword is gone; all callbacks are
  passed via `callback=` as standard `DiscreteCallback`s or a `CallbackSet`.

### Problem specification

- `MDCProblem(cost, p0, dp0, momentum, tspan)` →
  `MDCProblem(cost, θ₀, dθ₀, H; names=...)` plus `MDCSpan(negative, positive)`
  passed to `MDCSolve`. The timespan is no longer part of the problem; it is
  supplied at solve time. `MDCProblem` now requires `H > cost(θ₀)` (enforced
  in `initialise_lambda`).
- `MDCProblemJumpStart(...)` — **removed**. To start the curve away from `θ₀`,
  shift `θ₀` manually before constructing the problem.
- `curveProblem`, `specify_curve` — **removed** (already deprecated in v0.3.x).

### Cost functions

- `DiffCost(f, f2)` where `f2(p, g)` mutates `g` → `CostFunction(f, g!)` where
  `g!(g, θ)` mutates `g`. **Argument order is swapped**: v0.3.x used `(p, g)`,
  v0.4.0 uses `(g, θ)`.
- `CostFunction(f, g!, fg)` — new: optionally supply a combined `fg(g, θ)`
  that computes value and gradient in one call. The solver hot loop uses
  `fg` via `value_and_gradient!` when supplied, halving the user-side call
  count per RHS evaluation when forward computation can be shared (e.g.
  adjoint-based gradients of simulation costs).
- `CostFunction(fg)` — new convenience constructor: supply only the combined
  `fg`; `f` and `g!` are derived from it.
- `value_and_gradient!(cost, g, θ)` — new method: combined value + gradient
  evaluation. Falls back to separate `gradient!` + `value` if `fg` is not
  supplied.
- `make_fd_differentiable(cost)` — **removed**. Supply a gradient yourself
  (ForwardDiff, Zygote, FiniteDiff, or manual).
- `sum_losses(lArray, p0)` — **removed**. Compose costs on the user side.
- `build_injection_loss(prob, solmethod, tpoints, output_map)` — **removed**.
  Build simulation-based costs yourself.

### Transforms

- `TransformationStructure(name, p_transform, inv_p_transform)` →
  `TransformChain(ts...)` of typed transforms (`ScaleTransform`,
  `LogAbsTransform`, `FixedParamsTransform`). Transforms are now composable
  and the chain handles gradient pullback automatically.
- `logabs_transform(p0)` → chain `LogAbsTransform()` with a `ScaleTransform`
  whose weights are the signs of the physical reference point:

```julia
signs = [p >= 0 ? 1 : -1 for p in p0]
chain = TransformChain(LogAbsTransform(), ScaleTransform(signs))
qqq

  This restores v0.3.x's sign-tracking behavior using the composable chain
  machinery. The `LogAbsTransform` itself remains a pure `exp`/`log(abs())`
  transform with no sign state; signs live on the chained `ScaleTransform`.
- `bias_transform(p0, indices, biases)` → `ScaleTransform(w)` with a full
  weights vector (use `1.0` for un-scaled components).
- `fix_params(p0, indices)` (derives fixed values from `p0`) →
  `FixedParamsTransform(free_idx, fixed_vals, full_dim)` (fixed values are
  explicit).
- `only_free_params(p0, indices)` — **removed as a named function**. Compute
  `fixed_idx = setdiff(1:full_dim, indices)` and use `FixedParamsTransform`.
- `transform_cost(cost, p0, tr)` returning `(DiffCost, newp0)` →
  `TransformedCost(cost, chain)` + `inverse(chain, θ₀)` for the transformed
  initial point.
- `transform_problem(prob, tr)`, `transform_ODESystem(od, tr)` — **removed**.
  The package no longer reparameterizes `ODEProblem` or `ODESystem` via
  ModelingToolkit. Users handle MTK transformations on their side and pass
  the resulting cost function to `MDCProblem`. ModelingToolkit is no longer a
  dependency.

### Callbacks

- `MomentumReadjustment(tol, verbose)` → `mdc_momentum_readjustment(sys; tol)`.
- `TerminalCond()` (always added implicitly) → `mdc_safety_callback(sys; tol)`
  (now opt-in; recommended as a default).
- `Verbose([CurveDistance(ts), HamiltonianResidual(ts)])` →
  `mdc_verbose_callbacks(sys, timepoints; is_negative=false)`. Returns a tuple
  of `PresetTimeCallback`s.
- `ParameterBounds(ids, lbs, ubs)` → `mdc_bounds_callback(ids, lbs, ubs)`.
- `StateReadjustment(tol, verbose)` — **removed**.
- The `mdc_callback` keyword of `evolve` is gone. Pass a `DiscreteCallback`,
  `CallbackSet`, or `nothing` to `MDCSolve` via `callback=`.

### Solution access

- `mdc.sol` (single `ODESolution`) → `curve.positive_sol` and
  `curve.negative_sol` (two separate `ODESolution`s, either may be `nothing`).
- `mdc(t)` returning `(states=..., costates=...)` →
  `curve(t; type=:all|:parameters|:costates)`.
- `trajectory(mdc)`, `trajectory(mdc, ts)` →
  `curve(t; type=:parameters)` or extract from `curve.positive_sol.u`.
- `costate_trajectory(mdc)` → `curve(t; type=:costates)`.
- `distances(mdc)` → `curve.positive_sol.t`, `curve.negative_sol.t`
  (concatenate manually).
- `Δ(mdc)` → `curve(t; type=:parameters) .- curve(0.0; type=:parameters)`.
- `cost_trajectory(mdc, ts)` → `cost_trajectory(curve, ts)` — restored as a
  function that recovers the cost via `curve.spec.cost`. No need to attach a
  cost function post-hoc as in v0.3.x's `add_cost(mdc, cost)`.
- `add_cost(mdc, cost)` — **removed**. The cost is always reachable via
  `curve.spec.cost`.
- `output_on_curve(f, mdc, t)` — **removed**. Use
  `f(curve(t; type=:parameters))` directly.
- `biggest_movers(mdc, n)` — **removed as a named function**. The plot recipe
  does top-mover filtering internally.

### Initialization

- `l2_hessian(nom_sol)` — **removed**. Compute the Hessian yourself.
- New: `sparse_init_dir(hessian; orthogonal_to, λ, ...) -> (v, val)` —
  soft-thresholded iterative eigenvector finder for sparse initial directions.
- New: `sparse_eigenbasis(hessian, num_vectors; λ, ...) -> (basis, values)` —
  sequential sparse eigenvector finder with orthogonality constraints.

### Plotting

- The `@recipe` for `MDCSolution` is significantly extended: new keywords
  `max_lines`, `mode` (`:absolute`/`:relative`), `raw` (transformed vs physical
  space), `density`.
- `output_on_curve(f, mdc, t)` → `animate_mdc(curve, user_sim_func; ...)` in the
  `MDCPlotsExt` extension (loaded when `Plots` is available). Produces a
  multi-panel animation: parameter traces, instantaneous deltas, and a
  user-supplied simulation plot.


## Performance and quality posture (new in v0.4.0)

- Zero-allocation vector field hot path with pre-generated `fwd_caches`.
- The 4-argument `TransformedCost` callable `(θ, gθ, gz, buffers)` is
  allocation-free when the underlying cost and transform chain are
  allocation-free.
- `value_and_gradient!` dispatch is resolved at compile time via type
  parameters; the `Nothing` case (no `fg` supplied) incurs no runtime branch.
- JET.jl type-stability and runtime-error QA tests in `test/qa/`.
- `AllocCheck` allocation tests in `test/alloc_tests.jl`.
- `PrecompileTools` workload in `src/precompilation.jl` covering common call
  paths, including a small end-to-end solve.


## Removed dependencies

- `ModelingToolkit` — no longer a dependency. v0.3.x's `transform_ODESystem` /
  `transform_problem` required it; v0.4.0 does not reparameterize ODE systems.
  Users who need MTK-based reparameterization can do it on their side with
  current MTK versions.
- `FiniteDiff` — no longer a dependency. Use `ForwardDiff` (or your preferred
  AD) on the user side.
- `ThreadsX` — replaced by `Threads.@spawn` in `MDCSolve`.


## Minimum supported versions

- Julia 1.10+
- OrdinaryDiffEq, DiffEqCallbacks, SciMLBase, PrecompileTools, RecipesBase,
  LinearAlgebra — see `[compat]` in `Project.toml` for specific bounds.
