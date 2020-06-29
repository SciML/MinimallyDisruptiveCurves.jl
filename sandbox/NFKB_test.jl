using FiniteDiff
using DiffEqParamEstim
using MinimallyDisruptiveCurves
using LinearAlgebra
using OrdinaryDiffEq
# gr()


    
# include("../models/forced_mass_spring.jl")
# include("models/circadian_model2.jl")

# include("models/STG_Liu.jl")

heaviside = soft_heaviside(0.01,3600.)
od,ic,tspan,ps = MinimallyDisruptiveCurves.NFKBModel(heaviside)
prob = ODEProblem(od,ic,tspan,ps)


to_fix = ["c2c","c2","c2a","c3c", "c1c", "a2"]
tstrct_fix = fix_params(last.(ps), get_name_ids(ps, to_fix))
od, ic, ps = transform_problem(prob,tstrct_fix; unames = first.(ic), pnames = first.(ps))
prob = ODEProblem(od,ic,tspan,ps)
tstrct_log = logabs_transform(last.(ps))
od, ic, ps = transform_problem(prob, tstrct_log; unames=first.(ic), pnames = first.(ps))


prob = ODEProblem(od,ic,tspan,ps)
tsteps = range(0, stop=tspan[end], length=100)
nomsol = solve(prob,Tsit5(), saveat=tsteps)
data = hcat(nomsol(tsteps)...)

om = MinimallyDisruptiveCurves.NFKB_output_map
l2l(sol) = sum([sum(abs2, el) for el in (om.((sol(tsteps).u)) .- om.((nomsol(tsteps).u)))])
L2L = L2Loss(tsteps, data)
cost = build_loss_objective(prob, Tsit5(), l2l; mpg_autodiff=true)

p0 = prob.p


using DiffEqSensitivity
function l2hess(prob)
   
    sp = ODEForwardSensitivityProblem(prob.f, prob.u0, prob.tspan, prob.p)
    psol = solve(sp, DP8())  
    x, dp = extract_local_sensitivities(psol)
    
    #ie multiply derivatives by output_map. Latter is parameter invaraint so this is the product rule (don't need a dom/dp*x term)
    dou = [hcat([om(col) for col in eachcol(el)]...) for el in dp]
 
   #sum over i,t dyidpj 
   dydp(i,j,k) = dp[i]
   n = size(dou[1])[1]
   q = length(prob.p)
   hess = zeros(q,q)
    for i = 1:q
        for j = 1:q
            for o = 1:n
                     hess[i,j] += dou[i][o,end]*dou[j][o,end]
            end
        end
    end
   return hess
end

function l2hessf()
    function predict(p)
        pprob = remake(prob; p = eltype(p).(p), u = eltype(p).(prob.u0) )
        solve(prob, Tsit5())
    end
    g = ForwardDiff.jacobian(predict, p0)
end


# hess = FiniteDiff.finite_difference_hessian(cost, p0)
hess = l2hess(prob)

F = eigen(hess)
dθ₀ = F.vectors[:,1]


mom = 2.
eprob = curveProblem(cost, p0, dθ₀; momentum=mom, span= (0.,1.))
cb1 = MomentumReadjustment(cost, eprob; momentum = mom, tol = 1e-3)
cb2 = VerboseOutput(:low, 0.1:0.1:10)
# cb3 = ParameterBounds([1,2,3], [0.,0.,0.], [10.,10.,10.])
cb = CallbackSet(cb1,cb2)
mdc = evolve(eprob, Tsit5; callback=cb, maxiters=1000)

