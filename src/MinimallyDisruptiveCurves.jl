module MinimallyDisruptiveCurves

using DiffEqBase, DiffEqCallbacks, FiniteDiff, LinearAlgebra, ModelingToolkit, Optim, ForwardDiff


import ModelingToolkit: modelingtoolkitize


include("utilities/loss_algebra.jl")
include("utilities/extra_loss_functions.jl")
include("utilities/helper_functions.jl")
include("utilities/solution_parsing.jl")
include("utilities/transform_structures.jl")
include("initial_direction.jl")
include("evolve_options.jl")
include("evolve.jl")



# extend the following base functions
import Base: show

export curveProblem, modelingtoolkitize

export TransformationStructure, logabs_transform, transform_problem, only_free_params, fix_params, sum_losses, build_injection_loss, get_name_ids, soft_heaviside

export MomentumReadjustment, StateReadjustment, VerboseOutput, ParameterBounds
end # module
