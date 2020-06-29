using ModelingToolkit
using OrdinaryDiffEq



"""
solves STG calcium model (Liu 1998)

The log(3000.0/Ca) in the algorithm doesn't play nicely with Zygote. Don't know why. I've maxed it with zero to make it work.
"""



function  CalciumNeuron(input)
    
    @parameters t
    @derivatives D'~t


    @parameters eNa eh eK eleak tauCa gNabar gCaSbar gCaTbar gKabar gKCabar gKdrbar ghbar gleak

    paramvars = [eNa, eh, eK, eleak, tauCa, gNabar, gCaSbar, gCaTbar, gKabar, gKCabar, gKdrbar, ghbar, gleak]

    @variables V(t) Ca(t) mNa(t) hNa(t) mCaS(t) hCaS(t) mCaT(t) hCaT(t) mIh(t) mKa(t) hKa(t) mKCa(t) mKdr(t) 
    statevars = [V, Ca, mNa, hNa, mCaS, hCaS, mCaT, hCaT, mIh, mKa, hKa, mKCa, mKdr]
    
    # calcium reversal potential
    eCa = (500.0)*(8.6174e-5)*(283.15)*(log(max((3000.0/Ca),1e-5)))




    # currents
    INa=  gNabar*mNa^3*hNa*(eNa - V)
    ICaS= gCaSbar*mCaS^3*hCaS*(eCa - V)
    ICaT= gCaTbar*mCaT^3*hCaT*(eCa - V)
    Ih= ghbar*mIh*(eh - V)
    IKa= gKabar*mKa^3*hKa*(eK - V)
    IKCa=  gKCabar*mKCa^4*(eK - V)
    IKdr=  gKdrbar*mKdr^4*(eK - V)
    Ileak= gleak*(eleak - V)


    # calcium
    Ca_inf = 0.05 + 0.94*(ICaS + ICaT);

    # gating
    Na_m_inf=  1.0./(1.0+exp((V+25.5)./-5.29));
    Na_h_inf=  1.0./(1.0+exp((V+48.9)./5.18));
    Na_tau_m=  1.32 - 1.26./(1+exp((V+120.0)./-25.0));
    Na_tau_h=  (0.67./(1.0+exp((V+62.9)./-10.0))).*(1.5+1.0./(1.0+exp((V+34.9)./3.6)));

    CaS_m_inf=  1.0./(1.0+exp((V+33.0)./-8.1));
    CaS_h_inf=  1.0./(1.0+exp((V+60.0)./6.2));
    CaS_tau_m=  1.4 + 7.0./(exp((V+27.0)./10.0) + exp((V+70.0)./-13.0));
    CaS_tau_h=  60.0 + 150.0./(exp((V+55.0)./9.0) + exp((V+65.0)./-16.0));

    CaT_m_inf=  1.0./(1.0 + exp((V+27.1)./-7.2));
    CaT_h_inf=  1.0./(1.0 + exp((V+32.1)./5.5));
    CaT_tau_m=  21.7 - 21.3./(1.0 + exp((V+68.1)./-20.5));
    CaT_tau_h=  105.0 - 89.8./(1.0 + exp((V+55.0)./-16.9));

    Ih_m_inf=  1.0./(1.0+exp((V+70.0)./6.0));
    Ih_tau_m=  (272.0 + 1499.0./(1.0+exp((V+42.2)./-8.73)));

    Ka_m_inf=  1.0./(1.0+exp((V+27.2)./-8.7));
    Ka_h_inf=  1.0./(1.0+exp((V+56.9)./4.9));
    Ka_tau_m=  11.6 - 10.4./(1.0+exp((V+32.9)./-15.2));
    Ka_tau_h=  38.6 - 29.2./(1.0+exp((V+38.9)./-26.5));

    KCa_m_inf=  (Ca./(Ca+3.0))./(1.0+exp((V+28.3)./-12.6));
    KCa_tau_m=  90.3 - 75.1./(1.0+exp((V+46.0)./-22.7));

    Kdr_m_inf=  1.0./(1.0+exp((V+12.3)./-11.8));
    Kdr_tau_m=  7.2 - 6.4./(1.0+exp((V+28.3)./-19.2));

    eqs = [ 
    D(V) ~              INa + ICaS + ICaT + Ih + IKa + IKCa + IKdr + Ileak,
    D(Ca) ~                   (1/tauCa)*(Ca_inf - Ca),
    D(mNa) ~                   (1/Na_tau_m)*(Na_m_inf - mNa), 
    D(hNa) ~                   (1/Na_tau_h)*(Na_h_inf - hNa), 
    D(mCaS) ~                   (1/CaS_tau_m)*(CaS_m_inf - mCaS), 
    D(hCaS) ~                   (1/CaS_tau_h)*(CaS_h_inf - hCaS), 
    D(mCaT) ~                   (1/CaT_tau_m)*(CaT_m_inf - mCaT), 
    D(hCaT) ~                   (1/CaT_tau_h)*(CaT_h_inf - hCaT), 
    D(mIh) ~                   (1/Ih_tau_m)*(Ih_m_inf - mIh), 
    D(mKa) ~                  (1/Ka_tau_m)*(Ka_m_inf - mKa), 
    D(hKa) ~                   (1/Ka_tau_h)*(Ka_h_inf - hKa), 
    D(mKCa) ~                  (1/KCa_tau_m)*(KCa_m_inf - mKCa), 
    D(mKdr) ~                   (1/Kdr_tau_m)*(Kdr_m_inf - mKdr)
    ]

    ps = paramvars .=> [50,-20,-80,-50,20,100,3,1.3,5,10,20,0.5,0.01]
    ics = statevars .=> [-60,0.05,0,0,0,0,0,0,0,0,0,0,0]

    od = ODESystem(eqs, t, statevars, paramvars)
    tspan = (0.,500.)
    return od, ics, tspan, ps
end



# function predict(p)
#     return Array(solve(prob, Tsit5(), p=p, saveat=tsteps, sensitivity=BacksolveAdjoint(;autojacvec=true)))        
# end


# function loss(p)
#     prediction = predict(p)
#     return sum(abs2, prediction - data)
# end

