module MinimallyDisruptiveCurves


using RecipesBase: RecipesBase, @recipe, @series
using LinearAlgebra: LinearAlgebra, dot, norm
using OrdinaryDiffEq: DiscreteCallback, ODEProblem, Tsit5
import OrdinaryDiffEq: solve

# 2. Grab the specific callback utilities from their native packages
using DiffEqCallbacks: PresetTimeCallback
using SciMLBase: terminate!

include("transforms.jl")
include("costInterface.jl")
include("MDCSystem.jl")
include("plotting_utilities.jl")

import Base.show

export MDCsolve
export AbstractCost, CostFunction, TransformedCost, inverse, forward, gradient!
export AbstractTransform, TransformChain, ScaleTransform, LogAbsTransform, FixedParamsTransform, OnlyFreeParamsTransform
export MDCSystem, MDCWorkspace, vectorfield, ODEProblem, MDCSpan, cost_profile

export mdc_safety_callback, mdc_bounds_callback, mdc_verbose_callbacks
export mdc_dHdu_residual, mdc_momentum_readjustment

export animate_mdc
function animate_mdc end

# Precompilation workload (must be at the end)
include("precompilation.jl")
end # module
