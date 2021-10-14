"""
f unctionality for working out an initial direction, given an abstract loss objective
"""

"""
    initial_costate(dθ₀, H, C₀)
arguments are initial curve direction, momentum, cost at initial parameters.
solves for the initial costate required to evolve a MD curve.
"""
function initial_costate(dθ₀, H, C₀)
    # H is the final cost/momentum, C₀ is initial cost at dθ₀
    μ₂ = (-H + C₀) / 2.
    λ₀ = -2. * μ₂ * dθ₀ 
    return λ₀
end

function initial_costate(c::MDCProblem)
    μ₂ = (-c.momentum + c.cost(c.p0)) / 2.
    λ₀ = -2. * μ₂ * c.dp0 
    return λ₀
end

function initial_conditions(c::MDCProblem)
    λ₀ = initial_costate(c)
    return cat(c.p0, λ0, dims=1)
end

"""
    l2_hessian(nom_sol)
gets hessian according to L2 loss under the assumption that loss(θ₀) = 0. nom_sol is the solution of the nominal ODEProblem. 
The Hessian then only requires first derivatives: it is sum_ij dyi/dθ * dyj/dtheta
"""
function l2_hessian(nom_sol)
    prob = nom_sol.prob
    function pToL2(p)
        pprob = remake(prob, p=p)
        psol = solve(pprob, nom_sol.alg, saveat=nom_sol.t) |> Array
        psol = reshape(psol, 1, :)
        return psol
    end

    gr = ForwardDiff.jacobian(pToL2, prob.p)
    u, d, v = svd(gr)
    return v * diagm(d.^2) * v' 
    # = gr'*gr but a bit more accurate
end
