# # Large-Scale Demonstration: NFKB Parameter Identification
# 
# This script consolidates the complete workflow of using `MinimallyDisruptiveCurves.jl` 
# on a complex biological model. We build an NFKB signaling pathway using `ModelingToolkit.jl`, 
# define a simulation-based cost function, compute sparse sensitivities, and trace 
# minimally disruptive curves through the high-dimensional parameter space.
# We also sparsify the parameters involved in our curves, which we haven't done previously. Curves involving fewer parameters are easier to interpret.
# The model is taken from: Lipniacki, Tomasz, et al. "Mathematical model of NF-κB regulatory module." Journal of theoretical biology 228.2 (2004): 195-215.

using ModelingToolkit
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq
using SciMLStructures: Tunable, canonicalize, replace
using PreallocationTools
using ForwardDiff
using MinimallyDisruptiveCurves
using ModelingToolkit: SymbolicT
using Plots
using SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_values
using SciMLBase
using LinearAlgebra

# ## 1. Model Definition
# 
# We define the NFKB pathway components and connect them to a stimulatory input.
# The model tracks various cytoplasmic and nuclear species, exposing key observables.

const t = ModelingToolkit.t_nounits
const D = ModelingToolkit.D_nounits

@component function NFKBWithPort(; name)
    @variables begin
        IKKN(t) = 0.0
        IKKa(t) = 0.0
        IKKi(t) = 0.0
        IKKaIkBa(t) = 0.0
        IKKaIkBaNfKb(t) = 0.0
        NFkB(t) = 0.0
        NFkBn(t) = 0.0
        A20(t) = 0.0
        A20t(t) = 0.0
        IkBa(t) = 0.0
        IkBan(t) = 0.0
        IkBat(t) = 0.0
        IkBaNfKb(t) = 0.06
        IkBaNfKbn(t) = 0.0
        Cgent(t) = 0.0
        NFkBn_obs(t)
        IkBa_cyto_obs(t)
        A20t_obs(t)
        IKKtot_obs(t)
        IKKa_obs(t)
        IkBat_obs(t)
    end

    vars = SymbolicT[]
    push!(vars, IKKN); push!(vars, IKKa); push!(vars, IKKi); push!(vars, IKKaIkBa)
    push!(vars, IKKaIkBaNfKb); push!(vars, NFkB); push!(vars, NFkBn); push!(vars, A20)
    push!(vars, A20t); push!(vars, IkBa); push!(vars, IkBan); push!(vars, IkBat)
    push!(vars, IkBaNfKb); push!(vars, IkBaNfKbn); push!(vars, Cgent)
    push!(vars, NFkBn_obs); push!(vars, IkBa_cyto_obs); push!(vars, A20t_obs)
    push!(vars, IKKtot_obs); push!(vars, IKKa_obs); push!(vars, IkBat_obs)

    @parameters begin
        kprod = 2.5e-5; kdeg = 0.000125; k1 = 0.0025; k2 = 0.1; k3 = 0.0015
        a1 = 0.5; a2 = 0.2; a3 = 1.0; t1 = 0.1; t2 = 0.1; c6a = 2.0e-5
        i1 = 0.0025; kv = 5.0; c1 = 5.0e-7; c3 = 0.0004; c4 = 0.5
        c5 = 0.0003; c4a = 0.5; c5a = 0.0001; i1a = 0.001; e1a = 0.0005
        c1a = 5.0e-7; c3a = 0.0004; e2a = 0.01

        c2 = 0.0, [tunable = false]
        c2a = 0.0, [tunable = false]
        c1c = 5.0e-7, [tunable = false]
        c2c = 0.0, [tunable = false]
        c3c = 0.0004, [tunable = false]
    end

    params = SymbolicT[]
    push!(params, kprod); push!(params, kdeg); push!(params, k1); push!(params, k2); push!(params, k3)
    push!(params, a1); push!(params, a2); push!(params, a3); push!(params, t1); push!(params, t2); push!(params, c6a)
    push!(params, i1); push!(params, kv); push!(params, c1); push!(params, c3); push!(params, c4)
    push!(params, c5); push!(params, c4a); push!(params, c5a); push!(params, i1a); push!(params, e1a)
    push!(params, c1a); push!(params, c3a); push!(params, e2a); push!(params, c2); push!(params, c2a)
    push!(params, c1c); push!(params, c2c); push!(params, c3c)

    remf_input = RealInput(; name = :remf_input)

    eqs = Equation[]
    push!(eqs, D(IKKN) ~ kprod - kdeg * IKKN - k1 * IKKN * remf_input.u)
    push!(eqs, D(IKKa) ~ -k3 * IKKa - kdeg * IKKa - a2 * IKKa * IkBa + t1 * IKKaIkBa - a3 * IKKa * IkBaNfKb + t2 * IKKaIkBaNfKb + (k1 * IKKN - k2 * IKKa * A20) * remf_input.u)
    push!(eqs, D(IKKi) ~ k3 * IKKa - kdeg * IKKi + k2 * IKKa * A20 * remf_input.u)
    push!(eqs, D(IKKaIkBa) ~ a2 * IKKa * IkBa - t1 * IKKaIkBa)
    push!(eqs, D(IKKaIkBaNfKb) ~ a3 * IKKa * IkBaNfKb - t2 * IKKaIkBaNfKb)
    push!(eqs, D(NFkB) ~ c6a * IkBaNfKb - a1 * NFkB * IkBa + t2 * IKKaIkBaNfKb - i1 * NFkB)
    push!(eqs, D(NFkBn) ~ i1 * kv * NFkB - a1 * IkBan * NFkBn)
    push!(eqs, D(A20) ~ c4 * A20t - c5 * A20)
    push!(eqs, D(A20t) ~ c2 + c1 * NFkBn - c3 * A20t)
    push!(eqs, D(IkBa) ~ -a2 * IKKa * IkBa - a1 * IkBa * NFkB + c4a * IkBat - c5a * IkBa - i1a * IkBa + e1a * IkBan)
    push!(eqs, D(IkBan) ~ -a1 * IkBan * NFkBn + i1a * kv * IkBa - e1a * kv * IkBan)
    push!(eqs, D(IkBat) ~ c2a + c1a * NFkBn - c3a * IkBat)
    push!(eqs, D(IkBaNfKb) ~ a1 * IkBa * NFkB - c6a * IkBaNfKb - a3 * IKKa * IkBaNfKb + e2a * IkBaNfKbn)
    push!(eqs, D(IkBaNfKbn) ~ a1 * IkBan * NFkBn - e2a * kv * IkBaNfKbn)
    push!(eqs, D(Cgent) ~ c2c + c1c * NFkBn - c3c * Cgent)
    push!(eqs, NFkBn_obs ~ NFkBn)
    push!(eqs, IkBa_cyto_obs ~ IkBa + IkBaNfKb)
    push!(eqs, A20t_obs ~ A20t)
    push!(eqs, IKKtot_obs ~ IKKN + IKKa + IKKi)
    push!(eqs, IKKa_obs ~ IKKa)
    push!(eqs, IkBat_obs ~ IkBat)

    sub_systems = System[]
    push!(sub_systems, remf_input)

    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()

    return ODESystem(
        eqs, t, vars, params;
        name = name,
        systems = sub_systems,
        initial_conditions = initial_conditions,
        guesses = guesses
    )
end

@component function NetworkSystem(; name)
    pathway = NFKBWithPort(; name = :pathway)

    t_switch = 3600.0
    steepness = 0.1
    height = 1.0

    eqs = Equation[]
    push!(eqs, pathway.remf_input.u ~ height / (1.0 + exp(-steepness * (t - t_switch))))

    sub_systems = System[]
    push!(sub_systems, pathway)

    return ODESystem(
        eqs, t, SymbolicT[], SymbolicT[];
        name = name,
        systems = sub_systems,
        initial_conditions = Dict{SymbolicT, SymbolicT}(),
        guesses = Dict{SymbolicT, SymbolicT}()
    )
end

build_nfkb() = NetworkSystem(; name = :env) |> structural_simplify

# ## 2. Setup MTK Network & Base ODE Problem
# 
# We instantiate the system, define the time span, and generate the baseline 
# experimental data that will serve as the target for our cost function.

sys = build_nfkb()
tspan = (0.0, 3600.0)
prob = ODEProblem(sys, [], tspan)

target_observables = [
    sys.pathway.NFkBn_obs,
    sys.pathway.IkBa_cyto_obs,
    sys.pathway.A20t_obs,
    sys.pathway.IKKtot_obs,
    sys.pathway.IKKa_obs,
    sys.pathway.IkBat_obs,
]

# ## 3. Dynamic Parameter Identification
# We identify the tunable parameters from the system and generate the nominal truth data.

params_to_optimize = tunable_parameters(sys) ∩ parameters(sys)

println("Natively optimizing $(length(params_to_optimize)) tunable parameters.")

timesteps = 0.0:10.0:3600.0
sol_nominal = solve(prob, Tsit5(); saveat = timesteps)
truth_data = Array(sol_nominal(timesteps, idxs = target_observables))

# ## 4. High-Performance Loss Function
# 
# We define an allocation-friendly loss function that uses PreallocationTools 
# and SciMLStructures to efficiently swap out parameter values and propagate 
# Dual numbers through the ODE solver for automatic differentiation.

function loss_function(x, p_tuple)
    odeprob, ts, truth, setter, diffcache, obs_symbols = p_tuple

    ps = parameter_values(odeprob)
    buffer = get_tmp(diffcache, x)

    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps_updated = replace(Tunable(), ps, buffer)
    setter(ps_updated, x)

    newprob = remake(odeprob; p = ps_updated)
    sol = solve(newprob, Tsit5(); saveat = ts)

    if sol.retcode != SciMLBase.ReturnCode.Success
        return eltype(x)(Inf)
    end

    current_data = sol(ts, idxs = obs_symbols)
    return sum(abs2, truth .- current_data) / length(truth)
end

setter = setp(prob, params_to_optimize)
getter = getp(prob, params_to_optimize)

raw_ps = parameter_values(prob)
tunable_vector_prototype = copy(canonicalize(Tunable(), raw_ps)[1])
diffcache = DiffCache(tunable_vector_prototype)

p_tuple = (prob, timesteps, truth_data, setter, diffcache, target_observables)

# ## 5. Evaluation and Gradient Verification
# 
# We set up the ForwardDiff configurations, wrap the cost and gradient into the 
# `CostFunction` structure, and apply a `LogAbsTransform` to explore parameter 
# sensitivities in relative (log) space.

println("\n--- Running Scaled Loss Function Evaluation ---")

x_nominal = getter(prob)
loss_at_nominal = loss_function(x_nominal, p_tuple)

f_wrapped = θ -> loss_function(θ, p_tuple)

x_nominal = getter(prob)
cfg = ForwardDiff.GradientConfig(f_wrapped, x_nominal, ForwardDiff.Chunk(x_nominal))

grad_wrapped! = function (g, θ)
    return ForwardDiff.gradient!(g, f_wrapped, θ, cfg)
end

base_cost = CostFunction(f_wrapped, grad_wrapped!)
pipeline = TransformChain(LogAbsTransform())
final_cost = TransformedCost(base_cost, pipeline)
x_nominal_transformed = MinimallyDisruptiveCurves.inverse(pipeline, x_nominal)

# ## 6. Sparse Eigenbasis and MDC Execution
# 
# We compute the Hessian at the nominal point to determine the initially insensitive directions.
# Rather than initialising along the raw eigenvectors (which are pretty dense, involving lots of nonzero parameter directions), we sparsity
# The λ parameter is the degree of sparsification. Higher λ means sparser initial directions

hess0 = ForwardDiff.hessian(θ -> final_cost(θ), x_nominal_transformed)
vs, vals = sparse_eigenbasis(hess0, 6; λ = 0.0001)

mdc_curves = Dict{Int, Any}()

for i in 1:5
    println("--- Running MDC for index i = $i ---")

    _mdc_sys = MDCProblem(
        final_cost,
        x_nominal_transformed,
        vs[i],
        1.0;
        names = params_to_optimize .|> Symbol
    )

    stabiliser = mdc_momentum_readjustment(_mdc_sys; tol = 1.0e-3)
    my_pipeline = CallbackSet(stabiliser)

    @time curves_i = MDCSolve(_mdc_sys, span = MDCSpan(-10.0, 10.0); callback = my_pipeline)

    mdc_curves[i] = curves_i
end

# ## 7. Visualizing the MDC Trajectories
# 
# For each curve, we create a figure with  two subplots: the parameter trajectories on top, and the cost trajectory on the bottom.
# We use the built-in `cost_trajectory` accessor to evaluate the cost along a uniform grid.
# We only show the 5 biggest moving parameters to avoid crowding

t_grid = range(-10.0, 10.0, length=200)


function build_plot(i)
    p_params = plot(mdc_curves[i], max_lines=5)
    cost_vals = cost_trajectory(mdc_curves[i], t_grid)
    p_cost = plot(t_grid, cost_vals, label="Cost", xlabel="Arc Length", ylabel="Cost", color=:black)
    return plot(p_params, p_cost, layout=(2, 1), size=(800, 600))
end;

# ## Trajectory 1
build_plot(1)
# ## Trajectory 2
build_plot(2)
# ## Trajectory 3
build_plot(3)
# ## Trajectory 4
build_plot(4)
# ## Trajectory 5
build_plot(5)



