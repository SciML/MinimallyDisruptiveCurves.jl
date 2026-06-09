using ModelingToolkit
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq
using SciMLStructures: Tunable, Constants, canonicalize, replace
using PreallocationTools
using ForwardDiff
using MinimallyDisruptiveCurves
using ModelingToolkit: SymbolicT

# Define independent variables and differential operators globally or within a module
const t = ModelingToolkit.t_nounits
const D = ModelingToolkit.D_nounits

@component function NFKBWithPort(; name)
    # 1. Macro form for unpacking names/default values
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

    # Push into concrete type-stable Vector{SymbolicT}
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

    # Explicitly catch the second reassignment of a2 if intended to be non-tunable
    # Note: Avoid declaring the exact same symbol multiple times in one block.
    # We assign metadata via the push or define it uniquely here.
    params = SymbolicT[]
    push!(params, kprod); push!(params, kdeg); push!(params, k1); push!(params, k2); push!(params, k3)
    push!(params, a1); push!(params, a2); push!(params, a3); push!(params, t1); push!(params, t2); push!(params, c6a)
    push!(params, i1); push!(params, kv); push!(params, c1); push!(params, c3); push!(params, c4)
    push!(params, c5); push!(params, c4a); push!(params, c5a); push!(params, i1a); push!(params, e1a)
    push!(params, c1a); push!(params, c3a); push!(params, e2a); push!(params, c2); push!(params, c2a)
    push!(params, c1c); push!(params, c2c); push!(params, c3c)

    remf_input = RealInput(; name = :remf_input)

    # 2. Main differential and observed equations as Vector{Equation}
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

    # Append the observed mappings into the main equations vector
    push!(eqs, NFkBn_obs ~ NFkBn)
    push!(eqs, IkBa_cyto_obs ~ IkBa + IkBaNfKb)
    push!(eqs, A20t_obs ~ A20t)
    push!(eqs, IKKtot_obs ~ IKKN + IKKa + IKKi)
    push!(eqs, IKKa_obs ~ IKKa)
    push!(eqs, IkBat_obs ~ IkBat)

    # Concrete collections for sub-systems
    sub_systems = System[]
    push!(sub_systems, remf_input)

    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()

    # ODESystem acts as the underlying wrapper for continuous models
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

    # Pure algebraic numeric constants (not parameters)
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

# Environment compiler entry-point
build_nfkb() = NetworkSystem(; name = :env) |> structural_simplify
