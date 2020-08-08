"""
The two stage method in DiffEqParamEstim is great, but it employs (intelligent) estimation of the derivative of the data over time. If we actually have a nominal solution, we know exactly what the derivative is over time. So this exploits it
"""


"""
For a prob::ODEProblem, with nominal parameters p0, creates a cost function C(p) 

C(p) is the two stage method (collocation) cost associated with  remake(prob::ODEProblem; p=p), described in the DiffEqParamEstim.jl docs. In this case, the 'data' is solve(prob).
"""
function build_injection_loss(prob::ODEProblem, solmethod::T, tpoints) where T <: DiffEqBase.AbstractODEAlgorithm
    pdim = length(prob.u0)
    nom_sol = Array(solve(prob, solmethod, saveat=tpoints))
    n = length(tpoints)
    
    function cost(p)
        pprob = remake(prob, p=p)
        du_nom = similar(prob.u0, promote_type(eltype(prob.u0), eltype(p)))
        du_p = similar(pprob.u0, promote_type(eltype(pprob.u0), eltype(p)))
        c = 0.
        for i = 1:n
            prob.f(du_nom,nom_sol[:,i], prob.p, tpoints[i])
            pprob.f(du_p, nom_sol[:,i], p, tpoints[i])
            c += sum(abs2, du_nom .- du_p) 
        end
        return c
    end

    function cost2(p,g)
        g[:] = ForwardDiff.gradient(cost, p)
        return cost(p)
    end
    return DiffCost(cost, cost2)
end