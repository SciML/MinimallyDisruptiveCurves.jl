

"""
Implements the reaction network detailed in https://www.pnas.org/content/pnas/suppl/2003/05/27/1132112100.DC1/2112SuppText.pdf

Accompanying article is here https://www.pnas.org/content/100/12/7051
"""





function CircadianModel(input)
@parameters t
@derivatives D'~t
@variables  M_P(t) M_C(t) M_B(t) P_C(t) C_C(t) P_CP(t) C_CP(t) PC_C(t) PC_N(t) PC_CP(t) PC_NP(t) B_C(t) B_CP(t) B_N(t) B_NP(t) I_N(t)

@parameters k_1 k_2 k_3 k_4 k_5 k_6 k_7 k_8 K_AP K_AC K_IB k_dmb k_dmc  k_dmp k_dn  k_dnc K_d K_dp K_p  K_mB K_mC K_mP k_sB k_sC k_sP m n V_1B V_1C V_1P V_1PC V_2B V_2C V_2P V_2PC V_3B V_3PC V_4B V_4PC V_phos v_dBC v_dBN v_dCC v_dIN v_dPC v_dPCC v_dPCN v_mB v_mC v_mP v_sB v_sC v_sP

#also have v_phos = 0.4 in the paper

eqs =  
[ D(M_P) ~ v_sP*(B_N^n/(K_AP^n + B_N^n)) - v_mP*(M_P/(K_mP + M_P)) - k_dmp*M_P,
  D(M_C) ~ v_sC*B_N^n/(K_AC^n + B_N^n) - v_mC*(M_C/(K_mC + M_C)) - k_dmc*M_C,
  D(M_B) ~ v_sB*K_IB^m/(K_IB^m + B_N^m ) - v_mB*(M_B/(K_mB + M_B)) - k_dmb*M_B,
  D(P_C) ~ k_sP*M_P - V_1P*P_C/(K_p + P_C) + V_2P*P_CP/(K_dp + P_CP) + k_4*PC_C - k_3*P_C*C_C - k_dn*P_C,
  D(C_C) ~ k_sC*M_C - V_1C*C_C/(K_p + C_C) + V_2C*C_CP/(K_dp + C_CP) + k_4*PC_C - k_3*P_C*C_C - k_dnc*C_C,
  D(P_CP) ~ V_1P*P_C/(K_p+P_C) - V_2P*P_CP/(K_dp + P_CP) - v_dPC*P_CP/(K_d + P_CP) - k_dn*P_CP,
  D(C_CP) ~ V_1C*C_C/(K_p +C_C) - V_2C*C_CP/(K_dp+C_CP) - v_dCC*C_CP/(K_d + C_CP) - k_dn*C_CP,
  D(PC_C) ~ -V_1PC*PC_C/(K_p + PC_C) + V_2PC*PC_CP/(K_dp + PC_CP) - k_4*PC_C + k_3*P_C*C_C + k_2*PC_N - k_1*PC_C - k_dn*PC_C,
  D(PC_N) ~ -V_3PC*PC_N/(K_p + PC_N) + V_4PC*PC_NP/(K_dp + PC_NP) - k_2*PC_N + k_1*PC_C - k_7*B_N*PC_N + k_8*I_N - k_dn*PC_N,
  D(PC_CP) ~ V_1PC*PC_C/(K_p + PC_C) - V_2PC*PC_CP/(K_dp + PC_CP) - v_dPCC*PC_CP/(K_d + PC_CP) - k_dn*PC_CP,
  D(PC_NP) ~ V_3PC*PC_N/(K_p + PC_N) - V_4PC*PC_NP/(K_dp + PC_NP) - v_dPCN*PC_NP/(K_d + PC_NP) - k_dn*PC_NP,
  D(B_C) ~ k_sB*M_B - V_1B*B_C/(K_p + B_C) + V_2B*B_CP/(K_dp + B_CP) - k_5*B_C + k_6*B_N - k_dn*B_C,
  D(B_CP) ~ V_1B*B_C/(K_p + B_C) - V_2B*B_CP/(K_dp + B_CP) - v_dBC*B_CP/(K_d + B_CP) - k_dn*B_CP,
  D(B_N) ~ -V_3B*B_N/(K_p + B_N) + V_4B*B_NP/(K_dp + B_NP) + k_5*B_C - k_6*B_N - k_7*B_N*PC_N + k_8*I_N - k_dn*B_N,
  D(B_NP) ~ V_3B*B_N/(K_p + B_N) - V_4B*B_NP/(K_dp + B_NP) - v_dBN*B_NP/(K_d + B_NP) - k_dn*B_NP,
  D(I_N) ~ -k_8*I_N + k_7*B_N*PC_N - v_dIN*I_N/(K_d + I_N) - k_dn*I_N
  ]



tspan = (0.,100.)

ic = [M_P=> 2.391615385357426 ,M_C=> 1.7547090952244027 ,M_B=> 8.561088205594936 ,P_C=> 0.9030978254420003 ,C_C=> 3.5790996574043907 ,P_CP=> 0.11556396765994004 ,C_CP=> 0.6301918743084862 ,PC_C=> 0.8525514643541497 ,PC_N=> 0.09761528221027084 ,PC_CP=> 0.1755706805316031 ,PC_NP=> 0.06289128872787488 ,B_C=> 2.2764826266300875 ,B_CP=> 0.9063725609755553 ,B_N=> 1.763837030511116 ,B_NP=> 0.32028080421972577 ,I_N=> 0.024197413185517866 ]


ps = [k_1=>0.4, k_2=>0.2, k_3=>0.4, k_4=>0.2, k_5=>0.4, k_6=>0.2, k_7=>0.5, k_8=>0.1, K_AP=>0.7, K_AC=>0.6, K_IB=>2.2, k_dmb=>0.01, k_dmc => 0.01, k_dmp=>0.01, k_dn => 0.01, k_dnc=>0.12, K_d=>0.3, K_dp=>0.1, K_p => 0.1, K_mB=>0.4, K_mC=>0.4, K_mP=>0.31, k_sB=>0.12, k_sC=>1.6, k_sP=>0.6, m=>2, n=>4, V_1B=>0.5, V_1C=>0.6, V_1P=>0.4, V_1PC=>0.4, V_2B=>0.1, V_2C=>0.1, V_2P=>0.3, V_2PC=>0.1, V_3B=>0.5, V_3PC=>0.4, V_4B=>0.2, V_4PC=>0.1, v_dBC=>0.5, v_dBN=>0.6, v_dCC=>0.7, v_dIN=>0.8, v_dPC=>0.7, v_dPCC=>0.7, v_dPCN=>0.7, v_mB=>0.8, v_mC=>1.0, v_mP=>1.1, v_sB=>1.0, v_sC=>1.1, v_sP=>1.5]


od = ODESystem(eqs, t, first.(ic), first.(ps))
of = ODEFunction(od)
tspan = (0.,100.)
return od, ic, tspan, ps
end




