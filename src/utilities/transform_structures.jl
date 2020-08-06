

struct TransformationStructure{T<:Function,U<:Function}
    name::Union{String,Nothing}
    p_transform::T
    inv_p_transform::U
end


"""
returns TransformationStructure that flips the signs of negative parameters, and then does the transform p -> log(p).
"""
function logabs_transform(p0)
    name = "logabs"
    is_positive = convert.(Int64, p0 .>= 0)
    is_positive[is_positive .== 0] .= -1
    pos(p) = p.* is_positive 
    p_transform(p) = log.(p.*is_positive)
    inv_p_transform(logp) = exp.(logp).*is_positive

    return TransformationStructure(name, p_transform, inv_p_transform)
end

"""
returns TransformationStructure that fixes parameters[indices]
"""
function fix_params(p0, indices)
    name = "fix indices $indices"
    indices |> unique! |> sort!
    not_indices = setdiff(collect(1:length(p0)), indices)
    p0 = deepcopy(p0)
    function p_transform(p)
        return deleteat!(deepcopy(p), indices)
    end

    function inv_p_transform(p)
        out = [p[1] for el in p0]
        out[not_indices] .= p 
        out[indices] .= p0[indices]
        return out
    end
    return TransformationStructure(name, p_transform, inv_p_transform)
end

"""
returns TransformationStructure. params[indices] -> biases.*params[indices]
"""
function bias_transform(p0, indices, biases)
    name = "bias indices $indices"
    indices |> unique! |> sort!
    not_indices = setdiff(collect(1:length(p0)), indices)
    all_biases = ones(size(p0))
    all_biases[indices] = biases
    all_inv_biases = 1. ./all_biases
    
    function p_transform(p)
        return  p.*all_biases
    end

    function inv_p_transform(p)
        return p.*all_inv_biases
    end

    return TransformationStructure(name, p_transform, inv_p_transform)
end

"""
returns TransformationStructure that fixes **all but** parameters[indices]
"""
function only_free_params(p0, indices)
    name = "only free indices are $indices"
    indices |> unique! |> sort!
    not_indices = setdiff(collect(1:length(p0)), indices)
    return fix_params(p0, not_indices)
end

"""
    transform_cost(cost, p0, tr::TransformationStructure; unames=nothing, pnames=nothing)
    return DiffCost(new_cost, new_cost2), newp0

Given a cost function C(p), makes a new differentiable cost function D(q), where q = tr(p) and D(q) = C(p)
"""
function transform_cost(cost, p0, tr::TransformationStructure; unames=nothing, pnames=nothing)   
    newp0 = tr.p_transform(p0)
    jac = ForwardDiff.jacobian

    function new_cost(p)
        return p |> tr.inv_p_transform |> cost
    end

    function new_cost2(p,g)           
            orig_p = tr.inv_p_transform(p)
            orig_grad  = deepcopy(orig_p)
            val = cost(orig_p, orig_grad)
            g[:] = jac(tr.inv_p_transform, p)*orig_grad
        return val
    end
    return DiffCost(new_cost, new_cost2), newp0
end


"""
Reparameterises prob::ODEProblem via the transformation tr. so newprob.p = tr(p) is an equivalent ODEProblem to prob.p = p
"""
function transform_problem(prob::ODEProblem, tr::TransformationStructure; unames = nothing, pnames=nothing)
    @parameters t
    @derivatives D'~t

    newp0 = tr.p_transform(prob.p)
    unames === nothing && (unames = [Variable(:x,i)(t) for i in eachindex(prob.u0)])
    pnames === nothing && (pnames = [Variable(:p,i)() for i in eachindex(newp0)])
    
    pnames = transform_names(pnames, tr)

    vars = reshape([Variable(Symbol(unames[i]))(t)   for i in eachindex(prob.u0)], size(prob.u0))
    params = reshape([Variable(Symbol(pnames[i]))()  for i in eachindex(newp0)], size(newp0))

    lhs = [D(var) for var in vars]
    if DiffEqBase.isinplace(prob)
        rhs = similar(vars, Any)
        prob.f(rhs, vars, tr.inv_p_transform(params), t)
    else
        rhs = prob.f(vars, tr.inv_p_transform(params), t)
    end

    eqs = vcat([lhs[i] ~ rhs[i] for i in eachindex(prob.u0)]...)
    de = ODESystem(eqs,t,vec(vars),vec(params))
    return de, (vars .=> prob.u0), (params .=> newp0)
end

"""
Transforms a set of parameter names via the name provided in tr
The output names are ModelingToolkit.Variable types
# Example

nv = [m, c, k]
tr.name = log
returns the variables: [log(m)], log(c), log(k)]
"""
function transform_names(nv, tr::TransformationStructure)
    nv = tr.p_transform(nv)
    names = repr.(nv)
    nnv = [Variable(Symbol(names[i]))() for i in eachindex(names)]
    return nnv
end

"""
searches for the parameter indices corresponding to names.
ps is an array of names
"""
function get_name_ids(ps, names::Array{String,1})
    # can't do a single findall as this doesn't preserve ordering
    all_names = repr.(ps)
    ids = [first(findall(x -> x == names[i], all_names)) for (i,el) in enumerate(names)]
    return ids
end

"""
searches for the parameter indices corresponding to names.
ps is an array of pairs: names .=> vals
"""
function get_name_ids(ps::Array{Pair{T,U},1}, names::Array{String,1}) where T where U
    return get_name_ids(first.(ps), names)
end