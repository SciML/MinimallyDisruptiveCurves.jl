module MinimallyDisruptiveCurves

using DiffEqBase, DiffEqCallbacks, OrdinaryDiffEq
using FiniteDiff, LinearAlgebra, ModelingToolkit, ForwardDiff
using RecipesBase


# import ModelingToolkit: modelingtoolkitize

include("mdc_types.jl")
include("utilities/loss_algebra.jl")
include("utilities/extra_loss_functions.jl")
include("utilities/helper_functions.jl")
include("utilities/solution_parsing.jl")
include("utilities/transform_structures.jl")
include("initial_direction.jl")
include("evolve_options.jl")
include("evolve.jl")

# include("../models/circadian_model.jl")
# include("../models/NFKB.jl")
# include("../models/STG_Liu.jl")
# include("../models/forced_mass_spring.jl")



# extend the following base functions
import Base: show

export curveProblem, evolve, trajectory, costate_trajectory

export TransformationStructure, logabs_transform, transform_problem, only_free_params, fix_params, sum_losses, build_injection_loss, get_name_ids, soft_heaviside, biggest_movers, get_ids_names

export MomentumReadjustment, StateReadjustment, VerboseOutput, ParameterBounds
export MinimallyDisruptiveCurve, Δ, distances, trajectory, costate_trajectory, add_cost, cost_trajectory
end # module
