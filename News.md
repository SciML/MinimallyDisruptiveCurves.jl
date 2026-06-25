# MinimallyDisruptiveCurves.jl v0.4.0

## Overview

v0.4.0 is a ground-up rewrite of the public API. Code written against v0.3.x
will not work without modification, and in most cases there is no
one-for-one substitution: the conceptual model has changed, not just the
function names. If you are upgrading from v0.3.x, treat this as a new
package that happens to share a name, and expect to rewrite your call sites
rather than port them.

## Architecture

The package is now organized around five composable pieces:

1. **Transform chains** (`TransformChain`) — composable reparameterizations
   of the optimization space (`ScaleTransform`, `LogAbsTransform`,
   `FixedParamsTransform`). Each transform exposes `forward`, `inverse`, and
   `pullback!` so gradients flow correctly through the chain. The MDC traces
   through the transformed (optimizer) space; the user's cost is evaluated in
   physical space.

2. **Cost interface** (`AbstractCost`, `CostFunction`, `TransformedCost`) —
   user-supplied cost and gradient are wrapped in a `CostFunction` and then
   in a `TransformedCost`, which threads the user's gradient through the
   transform chain via pullback.

3. **Problem and solver** (`MDCProblem`, `MDCWorkspace`, `MDCSpan`,
   `MDCSolution`, `MDCSolve`) — `MDCProblem` packages the cost, initial
   point, initial direction, momentum budget, and parameter names. `MDCSolve`
   integrates the Hamiltonian vector field forwards and backwards in
   arc-length from `[θ₀; λ₀]` using `Tsit5`.

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

### Costs
- `CostFunction(f, grad!)` — value and gradient supplied separately
- `TransformedCost(cost, chain)` — wraps a cost with a transform chain
- `value(cost, θ)`, `gradient!(cost, g, θ)`

### Problem and solve
- `MDCProblem(cost, θ₀, dθ₀, H; names=...)`
- `MDCWorkspace(sys)`
- `MDCSpan(negative, positive)`
- `MDCSolve(sys; span, mode, dt, callback, parallel) -> MDCSolution`
- `curve(t; type=:all|:parameters|:costates)` — continuous interpolation
  across the split-span trajectory

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
- A `RecipesBase` recipe is defined for `MDCSolution`.
- `animate_mdc(curve, user_sim_func; ...)` is provided via the
  `MDCPlotsExt` extension, loaded when `Plots` is available.

## Migration from v0.3.x

[Fill in from memory — for each major v0.3.x entry point, note the v0.4.0
replacement or mark as removed with no direct equivalent. Even a partial list
helps. If you'd rather not maintain a migration table for an API that's been
rewritten, the paragraph above ("treat this as a new package") is enough.]

## Performance and quality posture

- The vector field returned by `vectorfield(sys)` is allocation-free on the
  ODE hot path when used with the 4-argument `TransformedCost` callable and
  pre-generated `fwd_caches`.
- JET.jl static-analysis tests in `test/qa/` verify type stability and
  absence of runtime errors on the core paths.
- `AllocCheck` tests in `test/alloc_tests.jl` verify zero allocation in the
  vector field and bounded allocation in the discrete-callback residual.
- A `PrecompileTools` workload in `src/precompilation.jl` precompiles the
  common call paths, including a small end-to-end solve.

## Minimum supported versions

- Julia 1.10+
[Add others if relevant — OrdinaryDiffEq version, etc.]
