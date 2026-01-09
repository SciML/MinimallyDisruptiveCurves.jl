module MinimallyDisruptiveCurves

using DiffEqCallbacks: DiffEqCallbacks, FunctionCallingCallback, PresetTimeCallback,
    SavedValues, SavingCallback
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra, I, diagm, dot, norm, svd
using ModelingToolkit: ModelingToolkit, Num, ODEFunction, ODEProblem, ODESystem,
    Variable, equations, independent_variable, modelingtoolkitize,
    parameters, structural_simplify, substitute, unknowns
using OrdinaryDiffEq: OrdinaryDiffEq, CallbackSet, DiscreteCallback, EnsembleProblem,
    EnsembleThreads, Tsit5, remake, solve, terminate!
using RecipesBase: RecipesBase, @recipe, @series
using SciMLBase: SciMLBase
using ThreadsX: ThreadsX

include("MDCTypes.jl")
include("MDCProblem.jl")
include("MDCProblemJumpstart.jl")
include("MDCSolution.jl")
include("plotting_utilities.jl")
include("utilities/loss_algebra.jl")
include("utilities/extra_loss_functions.jl")
include("utilities/helper_functions.jl")
include("utilities/solution_parsing.jl")
include("utilities/transform_structures.jl")
include("evolve_options.jl")
include("evolve.jl")

import Base.show

export DiffCost, make_fd_differentiable, l2_hessian

export CurveProblem, specify_curve, evolve, trajectory, costate_trajectory
export MDCProblem, MDCProblemJumpStart, JumpStart, jumpstart

export TransformationStructure, logabs_transform, bias_transform, transform_problem,
    transform_ODESystem, only_free_params, fix_params, transform_cost

export sum_losses, build_injection_loss, get_name_ids, soft_heaviside, biggest_movers,
    get_ids_names

export MomentumReadjustment, StateReadjustment, VerboseOutput, ParameterBounds,
    CurveInfoSnippet, CurveDistance, HamiltonianResidual, Verbose, TerminalCond,
    CallbackCallable

export Δ, distances, trajectory, costate_trajectory, add_cost, cost_trajectory,
    output_on_curve

# Precompilation workload (must be at the end)
include("precompilation.jl")

end # module
