module MinimallyDisruptiveCurves

using RecipesBase: RecipesBase, @recipe
using LinearAlgebra: LinearAlgebra, dot, norm, eigen, mul!, normalize!
using OrdinaryDiffEq: DiscreteCallback, ODEProblem, Tsit5
import OrdinaryDiffEq: solve

# 2. Grab the specific callback utilities from their native packages
using DiffEqCallbacks: PresetTimeCallback
using SciMLBase: terminate!

include("transforms.jl")
include("costInterface.jl")
include("MDCProblem.jl")
include("plotting_utilities.jl")
include("utilities.jl")

import Base.show

export AbstractTransform, TransformChain, ScaleTransform, LogAbsTransform, FixedParamsTransform
export AbstractCost, CostFunction, TransformedCost, inverse, forward, forward!, pullback!, gradient!, value, value_and_gradient!, cost_trajectory
export MDCSolve, MDCProblem, MDCSpan
export generate_fwd_caches


export mdc_safety_callback, mdc_bounds_callback, mdc_verbose_callbacks
export mdc_dHdu_residual, mdc_momentum_readjustment


"""
    animate_mdc(args...; kwargs...)

Plotting extension hook for animations of minimally disruptive curves.

This generic is extended by `MDCPlotsExt` when `Plots.jl` is loaded. Load
`Plots` before calling it; the base package intentionally provides no plotting
backend or fallback implementation.

# Arguments
- `args...`: plotting data accepted by the active extension.

# Keyword Arguments
- `kwargs...`: plotting options accepted by the active extension.

# Example
```julia
using MinimallyDisruptiveCurves, Plots
# animate_mdc(curve; filename = "curve.gif")
```
"""
function animate_mdc end
export animate_mdc

export sparse_init_dir, sparse_eigenbasis

# Precompilation workload (must be at the end)
include("precompilation.jl")
end # module
