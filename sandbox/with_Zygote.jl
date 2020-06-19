"""
We can create loss functions on DE solutions using Zygote. However it seems this is somewhat slow. Anyway, the process is detailed here.

Why is it slow? e.g. see here https://docs.sciml.ai/stable/analysis/sensitivity/

The automatic differentiation differentiates the entire output of the solution struct.  The lower level implementation in build_loss_objective works an order of magnitude faster, presumably because it doesn't? (Not quite sure why so much faster even for forward sensitivity, which should be the same)
"""


include("../models/STG_Liu.jl")


# jac=true makes compilation longer, but is potentially useful later...
od,ic,ps = create(t->0)
prob = ODEProblem(od,ic,tspan,ps; jac=true)
nomsol = solve(prob, Tsit5(), p=last.(ps), saveat=tsteps)
data = Array(nomsol)
p0 = last.(ps)


"""
Note autojacvec=true in the AD kwargs. This then uses the prob.f jacobian (which is instantiated from the jac=true above).
"""
function predict(p)
    return Array(solve(prob, Tsit5(), p=p, saveat=tsteps, sensitivity=BacksolveAdjoint(;autojacvec=true)))        
end

function loss(p)
    prediction = predict(p)
    return sum(abs2, prediction - data)
end

"""
This crawls
"""
@time Zygote.gradient(loss,p0)
