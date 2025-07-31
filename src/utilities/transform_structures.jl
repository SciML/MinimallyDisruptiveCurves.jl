
struct TransformationStructure{T <: Function, U <: Function}
    name::Union{String, Nothing}
    p_transform::T
    inv_p_transform::U
end

"""
returns TransformationStructure that flips the signs of negative parameters, and then does the transform p -> log(p).
"""
function logabs_transform(p0)
    name = "logabs_"
    is_positive = convert.(Int64, p0 .>= 0)
    is_positive[is_positive .== 0] .= -1
    pos(p) = p .* is_positive
    p_transform(p) = log.(p .* is_positive)
    inv_p_transform(logp) = exp.(logp) .* is_positive

    return TransformationStructure(name, p_transform, inv_p_transform)
end

"""
returns TransformationStructure that fixes parameters[indices]
"""
function fix_params(p0, indices)
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
    return TransformationStructure(nothing, p_transform, inv_p_transform)
end

"""
returns TransformationStructure. params[indices] -> biases.*params[indices]
"""
function bias_transform(p0, indices, biases)
    indices |> unique! |> sort!
    not_indices = setdiff(collect(1:length(p0)), indices)
    all_biases = ones(size(p0))
    all_biases[indices] = biases
    all_inv_biases = 1.0 ./ all_biases

    function p_transform(p)
        return p .* all_biases
    end

    function inv_p_transform(p)
        return p .* all_inv_biases
    end

    return TransformationStructure(nothing, p_transform, inv_p_transform)
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
function transform_cost(
        cost, p0, tr::TransformationStructure; unames = nothing, pnames = nothing)
    newp0 = tr.p_transform(p0)
    jac = ForwardDiff.jacobian

    function new_cost(p)
        return p |> tr.inv_p_transform |> cost
    end

    function new_cost2(p, g)
        orig_p = tr.inv_p_transform(p)
        orig_grad = deepcopy(orig_p)
        val = cost(orig_p, orig_grad)
        g[:] = jac(tr.inv_p_transform, p) * orig_grad
        return val
    end
    return DiffCost(new_cost, new_cost2), newp0
end

"""
    new_od = transform_ODESystem(od::ODESystem, tr::TransformationStructure)

    - reparameterises the parameters of an ODE system via the transformation tr. 
    - within the ODE eqs, any instance of a parameter p is replaced with tr.inv_p_transform(p)
    - if there are default parameter values p0, they are changed to tr.p_transform(p0)
"""
function transform_ODESystem(od::ModelingToolkit.AbstractSystem, tr::TransformationStructure)
    t = independent_variable(od)
    unames = states(od) .|> Num
    eqs = equations(od)
    ps = parameters(od) .|> Num
    new_ps = transform_names(ps, tr) # modified names under transformation

    of = ODEFunction(od, eval_expression = false) # to solve world age issues
    rhs = similar(unames, eltype(unames))
    of(rhs, unames, tr.inv_p_transform(new_ps), t) # in place modification of rhs
    lhs = [el.lhs for el in eqs]

    _defaults = ModelingToolkit.get_defaults(od)

    # new_defaults -= map(_defaults) do (key, val)
    #     if k
    # end
    if length(_defaults) > 0
        p_collect = [el => _defaults[el] for el in ps] # in correct order
        if length(p_collect) > 0
            new_p_vals = tr.p_transform(last.(p_collect))
            new_p_dict = Dict(new_ps .=> new_p_vals)
            _defaults = merge(_defaults, new_p_dict)
        end
    end

    de = ODESystem(lhs .~ rhs, t, unames, new_ps, defaults = _defaults,
        checks = false, name = nameof(od))
    return de # (vars .=> last.(ic)), (new_ps .=> newp0)
end

"""
Reparameterises prob::ODEProblem via the transformation tr. so newprob.p = tr(p) is an equivalent ODEProblem to prob.p = p
"""

function transform_problem(prob::ODEProblem, tr::TransformationStructure; unames = nothing, pnames = nothing)
    println(pnames)
    sys = modelingtoolkitize(prob)
    eqs = ModelingToolkit.get_eqs(sys)
    pname_tr = parameters(sys) .=> pnames
    uname_tr = states(sys) .=> unames
    neweqs = eqs
    if !(pnames === nothing)
        neweqs = [el.lhs ~ substitute(el.rhs, pname_tr) for el in neweqs]
    else
        pnames = ModelingToolkit.get_ps(sys)
    end

    if !(unames === nothing)
        neweqs = [substitute(el.lhs, uname_tr) ~ substitute(el.rhs, uname_tr)
                  for el in neweqs]
    else
        unames = ModelingToolkit.get_states(sys)
    end
    named_sys = ODESystem(neweqs, ModelingToolkit.get_iv(sys), unames, pnames,
        defaults = merge(Dict(unames .=> prob.u0), Dict(pnames .=> prob.p)),
        name = nameof(sys))
    newp0 = tr.p_transform(prob.p)
    t_sys = transform_ODESystem(named_sys, tr)
    return t_sys, (ModelingToolkit.get_states(t_sys) .=> prob.u0),
    (ModelingToolkit.get_ps(t_sys) .=> newp0)
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
    if tr.name === nothing
        names = Symbol.(repr.(tr.p_transform(nv)))
    else
        names = Symbol.(tr.name .* repr.(nv))
    end
    new_vars = [Num(Variable(el)) for el in names]
    return new_vars
end

"""
searches for the parameter indices corresponding to names.
ps is an array of names
"""
function get_name_ids(ps, names::Array{String, 1})
    # can't do a single findall as this doesn't preserve ordering
    all_names = repr.(ps)
    ids = [first(findall(x -> x == names[i], all_names)) for (i, el) in enumerate(names)]
    return ids
end

"""
searches for the parameter indices corresponding to names.
ps is an array of pairs: names .=> vals
"""
function get_name_ids(ps::Array{Pair{T, U}, 1}, names::Array{String, 1}) where {T} where {U}
    return get_name_ids(first.(ps), names)
end
