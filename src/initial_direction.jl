"""
functionality for working out an initial direction, given an abstract loss objective

do we have to finite difference the hessian, or can we do a trick?

Note that we have a function loss(sol)
Now let's assume loss is zero at IC. then dLdsol = 0.
    So d2Lp = dLsol d2soldp + dsoldp d2ldsol dsoldp
    and the first term disappears.

But finite difference is much more robust and easier.
"""


function initial_costate(dθ₀, H, C₀)
    #H is the final cost/momentum, C₀ is initial cost at dθ₀
    μ₂ = (-H + C₀)/2
    λ₀ = -2*μ₂*dθ₀ 
    return λ₀
end

# function initial_md_direction(cost, θ₀;nth = 1, hessian = :none)
#     if hessian == :l2
        
#     else
#         hess = FiniteDiff.finite_difference_hessian(cost, θ₀)
#     end
#     F = eigen(hess)
#     nth = F.vectors[:,nth]
# end

# """
#     get minimally disruptive directions according to L2 loss under the assumption that loss(θ₀) = 0. The Hessian then only requires first derivatives, so is generically tractable. use the experimantal function second_order_sensitivitiesusing
# """
# function l2_hessian(nom_sol)
#     prob = nom_sol.prob
#     function pToL2(p)
#         pprob = remake(prob, p=p)
#         psol = solve(pprob, nom_sol.alg, saveat = nom_sol.t) |> Array
#         psol = reshape(psol, 1, :)
#         return psol
#     end

#     gr = ForwardDiff.jacobian(pToL2, prob.p)
#     return gr'*gr
# end
