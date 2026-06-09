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
include("MDCSystem.jl")
include("plotting_utilities.jl")
include("utilities.jl")

import Base.show

export AbstractTransform, TransformChain, ScaleTransform, LogAbsTransform, FixedParamsTransform, OnlyFreeParamsTransform
export AbstractCost, CostFunction, TransformedCost, inverse, forward, gradient!
export MDCSolve, MDCSystem, MDCWorkspace, vectorfield, ODEProblem, MDCSpan

export mdc_safety_callback, mdc_bounds_callback, mdc_verbose_callbacks
export mdc_dHdu_residual, mdc_momentum_readjustment


function animate_mdc end
export animate_mdc

export sparse_init_dir, sparse_eigenbasis

# Precompilation workload (must be at the end)
include("precompilation.jl")
end # module
