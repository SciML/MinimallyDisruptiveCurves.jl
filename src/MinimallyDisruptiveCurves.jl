module MinimallyDisruptiveCurves

using DiffEqBase, DiffEqCallbacks, OrdinaryDiffEq
using FiniteDiff, LinearAlgebra, ModelingToolkit, ForwardDiff
using RecipesBase


# import ModelingToolkit: modelingtoolkitize
include("MDCProblem.jl")
include("MDCSolution.jl")
include("utilities/loss_algebra.jl")
include("utilities/extra_loss_functions.jl")
include("utilities/helper_functions.jl")
include("utilities/solution_parsing.jl")
include("utilities/transform_structures.jl")
include("evolve_options.jl")
include("evolve.jl")

# include("../models/circadian_model.jl")
# include("../models/NFKB.jl")
# include("../models/STG_Liu.jl")
# include("../models/forced_mass_spring.jl")

export DiffCost, make_fd_differentiable, l2_hessian

export curveProblem, specify_curve, evolve, trajectory, costate_trajectory, MDCProblem

export TransformationStructure, logabs_transform, bias_transform, transform_problem, transform_ODESystem, only_free_params, fix_params, transform_cost

export sum_losses, build_injection_loss, get_name_ids, soft_heaviside, biggest_movers, get_ids_names

export MomentumReadjustment, StateReadjustment, VerboseOutput, ParameterBounds

export MinimallyDisruptiveCurve, Δ, distances, trajectory, costate_trajectory, add_cost, cost_trajectory
end # module
