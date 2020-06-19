"""
Try to make an ensemble problem corresponding to solution of the same ODE with a library of inputs. And get cost function etc working on it.
"""



include("models/mass_spring.jl")
t = collect(range(0,stop=10,length=100) )
# randomized = [(sol(t[i]) + .01randn(16)) for i in 1:length(t)]
# randomized = [sol(t[i])]
# d0ata = vecvec_to_mat(randomized)

stepf(t) = t<1 ? 1 : 0


nominal_inputs = [stepf, sin, cos]
nominal_vfs = [(δx,x,p,t) -> vf_forced(δx,x,p,t, inp) for inp in nominal_inputs]
nominal_problems = [ODEProblem(vf,u0,tspan,p0) for vf in nominal_vfs]
nominal_sols = [solve(prob,Tsit5()) for prob in nominal_problems]


function pf(prob,i,repeat)
    this_vf = (δx,x,p,t) -> vf_forced(δx,x,p,t, nominal_inputs[i])
    return ODEProblem(this_vf,u0,tspan,prob.p)
end


of(sol,i) = (sol,false)



ensemble_prob = EnsembleProblem(nominal_problems[1];
                prob_func= pf)
                # reduction = (u,data,I)->(append!(u,data),false),
                # u_init = [])

nom_ensemble_sol = sim = solve(ensemble_prob, Tsit5(); trajectories=length(nominal_inputs))



function lossf(sols)
    sum([sum(sols[i](t) - nominal_sols[i](t)) for i in 1:length(sols)])
end

b = build_loss_objective(ensemble_prob,Tsit5(),lossf, trajectories=3)