"""
The two stage method in DiffEqParamEstim is great, but it employs (intelligent) estimation of the derivative of the data over time. If we actually have a nominal solution, we know exactly what the derivative is over time. So this exploits it
"""

"""
For a prob::ODEProblem, with nominal parameters p0, creates a cost function C(p)
Let pprob = remake(prob, p=p).
C(p) is the collocation cost associated with  pprob. Calculated by integrating the following over the trajectory of solve(prob; saveat=tsteps):

int_u  sum(pprob.f(u) - prob.f(u)).^2

with output map g(x), this turns into

int_u  sum(dgdx(u)*pprob.f(u) - dgdx(u)*prob.f(u)).^2
"""
function build_injection_loss(prob::ODEProblem, solmethod::T, tpoints,
        output_map = x -> x) where {T <: DiffEqBase.AbstractODEAlgorithm}
    pdim = length(prob.u0)
    nom_sol = Array(solve(prob, solmethod, saveat = tpoints))
    n = length(tpoints)
    dgdx = x -> ForwardDiff.jacobian(output_map, x)
    dgdx_template = dgdx(prob.u0)
    dgdx_template2 = deepcopy(dgdx_template)

    function cost(p)
        pprob = remake(prob, p = p)
        du_nom = similar(prob.u0, promote_type(eltype(prob.u0), eltype(p)))
        du_p = similar(pprob.u0, promote_type(eltype(pprob.u0), eltype(p)))
        c = 0.0
        @inbounds for i in 1:n
            prob.f(du_nom, nom_sol[:, i], prob.p, tpoints[i])
            pprob.f(du_p, nom_sol[:, i], p, tpoints[i])
            dgdx_template = dgdx(nom_sol[:, i])
            c += sum(abs2, dgdx_template*du_nom .- dgdx_template*du_p)
        end
        return c
    end

    function cost2(p, g)
        g[:] = ForwardDiff.gradient(cost, p)
        return cost(p)
    end
    return DiffCost(cost, cost2)
end
