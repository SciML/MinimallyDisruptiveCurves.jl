using ModelingToolkit
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq
using SciMLStructures: Tunable, Constants, canonicalize, replace
using PreallocationTools
using ForwardDiff
using MinimallyDisruptiveCurves

# Define our independent variables globally or within a module
t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

@component function NFKBWithPort(; name)
    # 1. Declare ALL variables (both dynamic states and your observed outputs)
    vars = @variables begin
        # Dynamic states (with initial conditions)
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
        
        # New observed output variables (no initial conditions needed)
        NFkBn_obs(t)
        IkBa_cyto_obs(t)
        A20t_obs(t)
        IKKtot_obs(t)
        IKKa_obs(t)
        IkBat_obs(t)
    end
    
pars = @parameters begin
    kprod = 2.5e-5; kdeg = 0.000125; k1 = 0.0025; k2 = 0.1; k3 = 0.0015
    a1 = 0.5; a2 = 0.2; a3 = 1.0; t1 = 0.1; t2 = 0.1; c6a = 2.0e-5
    i1 = 0.0025; kv = 5.0; c1 = 5.0e-7; c3 = 0.0004; c4 = 0.5
    c5 = 0.0003; c4a = 0.5; c5a = 0.0001; i1a = 0.001; e1a = 0.0005
    c1a = 5.0e-7; c3a = 0.0004; e2a = 0.01
    
    # Explicitly mark these three as non-tunable constants
    a2 = 0.2, [tunable = false]
    c2 = 0.0, [tunable = false]
    c2a = 0.0, [tunable = false]
    c1c = 5.0e-7, [tunable = false]
    c2c = 0.0,    [tunable = false]
    c3c = 0.0004, [tunable = false]
end


    remf_input = RealInput(; name = :remf_input)
    
    # 2. Main differential equations
    eqs = [
        D(IKKN) ~ kprod - kdeg * IKKN - k1 * IKKN * remf_input.u
        D(IKKa) ~ -k3 * IKKa - kdeg * IKKa - a2 * IKKa * IkBa + t1 * IKKaIkBa - a3 * IKKa * IkBaNfKb + t2 * IKKaIkBaNfKb + (k1 * IKKN - k2 * IKKa * A20) * remf_input.u
        D(IKKi) ~ k3 * IKKa - kdeg * IKKi + k2 * IKKa * A20 * remf_input.u
        D(IKKaIkBa) ~ a2 * IKKa * IkBa - t1 * IKKaIkBa
        D(IKKaIkBaNfKb) ~ a3 * IKKa * IkBaNfKb - t2 * IKKaIkBaNfKb
        D(NFkB) ~ c6a * IkBaNfKb - a1 * NFkB * IkBa + t2 * IKKaIkBaNfKb - i1 * NFkB
        D(NFkBn) ~ i1 * kv * NFkB - a1 * IkBan * NFkBn
        D(A20) ~ c4 * A20t - c5 * A20
        D(A20t) ~ c2 + c1 * NFkBn - c3 * A20t
        D(IkBa) ~ -a2 * IKKa * IkBa - a1 * IkBa * NFkB + c4a * IkBat - c5a * IkBa - i1a * IkBa + e1a * IkBan
        D(IkBan) ~ -a1 * IkBan * NFkBn + i1a * kv * IkBa - e1a * kv * IkBan
        D(IkBat) ~ c2a + c1a * NFkBn - c3a * IkBat
        D(IkBaNfKb) ~ a1 * IkBa * NFkB - c6a * IkBaNfKb - a3 * IKKa * IkBaNfKb + e2a * IkBaNfKbn
        D(IkBaNfKbn) ~ a1 * IkBan * NFkBn - e2a * kv * IkBaNfKbn
        D(Cgent) ~ c2c + c1c * NFkBn - c3c * Cgent]
    
    # 3. Create your explicit observed mapping equations
    obs_eqs = [
        NFkBn_obs ~ NFkBn
        IkBa_cyto_obs ~ IkBa + IkBaNfKb
        A20t_obs ~ A20t
        IKKtot_obs ~ IKKN + IKKa + IKKi
        IKKa_obs ~ IKKa
        IkBat_obs ~ IkBat
    ]
    
    # 4. Pass the observed equations to the ODESystem constructor
    return ODESystem(
        eqs, t, vars, pars; 
        name = name, 
        systems = [remf_input], 
        observed = obs_eqs 
    )
end

@component function NetworkSystem(; name)
    # 1. Instantiate only your biological network
    pathway = NFKBWithPort(; name = :pathway)
    
    # 2. Define the smooth step configuration locally as standard variables
    # (By keeping them outside a @parameters block, they act as pure constants)
    t_switch  = 3600.0   # Activation timepoint
    steepness = 0.1      # Transition sharpness
    height    = 1.0      # Jump magnitude
    
    # 3. Use a regular algebraic equation instead of connect()
    eqs = [
        pathway.remf_input.u ~ height / (1.0 + exp(-steepness * (t - t_switch)))
    ]
    
    # 4. Only include the pathway system in the sub-systems array
    return ODESystem(eqs, t; name = name, systems = [pathway])
end

# 1. Compose the environment
build_nfkb() = NetworkSystem(; name = :env) |> structural_simplify
