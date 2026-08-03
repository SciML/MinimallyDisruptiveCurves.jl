# API Reference

## Cost Interface

```@docs
AbstractCost
CostFunction
TransformedCost
value
gradient!
value_and_gradient!
```

## Transformation Interface

```@docs
AbstractTransform
TransformChain
ScaleTransform
LogAbsTransform
FixedParamsTransform
forward
forward!
inverse
pullback!
generate_fwd_caches
```

## Curve Construction And Solution

```@docs
MDCProblem
MDCSpan
MDCSolve
cost_trajectory
```

## Callbacks And Utilities

```@docs
mdc_safety_callback
mdc_bounds_callback
mdc_verbose_callbacks
mdc_dHdu_residual
mdc_momentum_readjustment
sparse_init_dir
sparse_eigenbasis
animate_mdc
```

## Developer Interfaces

These contracts support package extensions. They are versioned and documented, but are
not intended as general application-level construction APIs.

```@docs
MinimallyDisruptiveCurves.MDCSolution
MinimallyDisruptiveCurves.transform_names
```
