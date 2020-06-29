



"""
This doesn't seem to be giving the right answer. Don't know why. The step function is either ill conditioned or doesn't kick the system like in the matlab simulation.
Solved. The heaviside function was ill conditioned. Made it a sigmoid and everything works
"""



function  NFKBModel(input)
    
@parameters t
@derivatives D'~t


@parameters kprod kdeg k1 k2 k3 a1 a2 a3 t1 t2 c6a i1 kv c1 c2 c3 c4 c5 c4a c5a i1a e1a c1a c2a c3a e2a c1c c2c c3c
paramvars = [kprod, kdeg, k1, k2, k3, a1, a2, a3, t1, t2, c6a, i1, kv, c1, c2, c3, c4, c5, c4a, c5a, i1a, e1a, c1a, c2a, c3a, e2a, c1c, c2c, c3c]

@variables IKKN(t) IKKa(t) IKKi(t) IKKaIkBa(t) IKKaIkBaNfKb(t) NFkB(t) NFkBn(t) A20(t) A20t(t) IkBa(t) IkBan(t) IkBat(t) IkBaNfKb(t) IkBaNfKbn(t) Cgent(t)
statevars = [IKKN, IKKa, IKKi, IKKaIkBa, IKKaIkBaNfKb, NFkB, NFkBn, A20, A20t, IkBa, IkBan, IkBat, IkBaNfKb, IkBaNfKbn, Cgent]



eqs = [ 
D(IKKN) ~              kprod - kdeg*IKKN - k1*IKKN*input(t),
D(IKKa) ~                  -k3*IKKa-kdeg*IKKa-a2*IKKa*IkBa + t1*IKKaIkBa - a3*IKKa*IkBaNfKb + t2*IKKaIkBaNfKb + (k1*IKKN - k2*IKKa*A20)*input(t), 
D(IKKi) ~                   k3*IKKa - kdeg*IKKi + k2*IKKa*A20*input(t), 
D(IKKaIkBa) ~                   a2*IKKa*IkBa - t1*IKKaIkBa, 
D(IKKaIkBaNfKb) ~                   a3*IKKa*IkBaNfKb - t2*IKKaIkBaNfKb, 
D(NFkB) ~                   c6a*IkBaNfKb - a1*NFkB*IkBa + t2*IKKaIkBaNfKb - i1*NFkB, 
D(NFkBn) ~                   i1*kv*NFkB - a1*IkBan*NFkBn, 
D(A20) ~                   c4*A20t - c5*A20, 
D(A20t) ~                   c2 + c1*NFkBn-c3*A20t, 
D(IkBa) ~                  -a2*IKKa*IkBa - a1*IkBa*NFkB + c4a*IkBat - c5a*IkBa - i1a*IkBa + e1a*IkBan, 
D(IkBan) ~                  -a1*IkBan*NFkBn + i1a*kv*IkBa-e1a*kv*IkBan, 
D(IkBat) ~                  c2a + c1a*NFkBn - c3a*IkBat, 
D(IkBaNfKb) ~                   a1*IkBa*NFkB - c6a*IkBaNfKb - a3*IKKa*IkBaNfKb + e2a*IkBaNfKbn, 
D(IkBaNfKbn) ~                   a1*IkBan*NFkBn - e2a*kv*IkBaNfKbn, 
D(Cgent) ~                   c2c + c1c*NFkBn - c3c*Cgent
]

ps = paramvars .=> [ 2.5e-5, 1.25e-4, 0.0025, 0.1, 0.0015, 0.5, 0.2, 1.0, 0.1, 0.1, 2.0e-5, 0.0025, 5.0, 5.0e-7, 0, 4.0e-4, 0.5, 3.0e-4, 0.5, 1.0e-4, 0.001, 5.0e-4, 5.0e-7, 0, 4.0e-4, 0.01, 5.0e-7, 0, 4.0e-4]

temp = zeros(15)
temp[13] = 0.06
tspan = (0., 50000.)
ic = statevars .=> temp

od = ODESystem(eqs, t, statevars, paramvars)
println("optional output map is [x[7], x[10] + x[13], x[9], x[1] + x[2] + x[3], x[2], x[12]]")
return od, ic, tspan, ps
end

function NFKB_output_map(x)
    return [x[7], x[10] + x[13], x[9], x[1] + x[2] + x[3], x[2], x[12]]
end
NFKB_output_map(x,t,integrator) = output_map(x)


# heaviside = soft_heaviside(0.01, 3600.)
# od0, ic0, tspan, ps0 = create(heaviside)
# prob = ODEProblem(od0, ic0, tspan, ps0)
# to_fix = ["c2c","c2","c2a","c3c", "c1c", "a2"]
# tstrct_fix = fix_params(last.(ps0), get_name_ids(ps0, to_fix))
# od2,ic2,ps2 = transform_problem(prob,tstrct_fix; unames = first.(ic0), pnames = first.(ps0))
# prob2 = ODEProblem(od2,ic2,tspan,ps2)
# tstrct_log = logabs_transform(last.(ps2))
# od3,ic3,ps3 = transform_problem(prob2,tstrct_log; unames = first.(ic2), pnames = first.(ps2))

# od,ic,ps = od3,ic3,ps3

# create() = (od,ic, tspan, ps)


# sol = solve(prob, Tsit5())
# plot(sol)






# saved_values = SavedValues(Float64, Array{Float64,1})
# output_callback = SavingCallback(output_map, saved_values)

# sol = solve(prob, AutoVern7(Rodas4()), callback = output_callback)



# c2,c2c,c2a,c3c,c1c,c5a