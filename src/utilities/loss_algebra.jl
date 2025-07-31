"""
Utilities to add cost functions together
"""

"""
A struct for holding differentiable cost functions. Let d::DiffCost.
d(p) returns cost at p
d(p,g) returns cost and **mutates** gradient at p
"""
struct DiffCost{F, F2} <: Function
    cost_function::F
    cost_function2::F2
end
(f::DiffCost)(x) = f.cost_function(x)
(f::DiffCost)(x, y) = f.cost_function2(x, y)

"""
given a cost function C(p), makes a new cost d::DiffCost, which has a method for returning the finite-difference gradient.
"""
function make_fd_differentiable(cost)
    function cost2(p, g)
        FiniteDiff.finite_difference_gradient!(g, cost, p)
        return cost(p)
    end
    return DiffCost(cost, cost2)
end

"""
    sum_losses(lArray::Array{T,1}, p0) where T <: Function

  - Given array of cost functions c1...cn, makes new cost function D. Multithreads evaluation of D(p) and D(p,g) if threads are available.
  - c1...cn must be differentiable: ci(p,g) returns cost and mutates gradient g

D(p) = sum_i c_i(p)
D(p,g) mutates g to give gradient of D.
"""
function sum_losses(lArray::Array{T, 1}, p0) where {T <: Function}
    (Threads.nthreads() == 1) &&
        (@info "Note that restarting julia with multiple threads will increase performance of the generated loss function from sum_losses()")
    dummy_gs = [deepcopy(p0) for el in lArray]::Vector{typeof(p0)}
    cs = [0.0 for el in lArray]
    function cost1(p)
        return sum(lArray) do loss
            loss(p)
        end
    end
    n = length(lArray)
    cs = Vector{Float64}(undef, n)
    pure_costs = map(lArray) do el
        (p, g) -> (el[i](p, g), g)
    end

    function cost2(p, g)
        ThreadsX.foreach(enumerate(lArray)) do (i, el)
            cs[i] = el(p, dummy_gs[i])
        end
        g[:] = sum(dummy_gs)
        return sum(cs)
    end
    return DiffCost(cost1, cost2)
end
