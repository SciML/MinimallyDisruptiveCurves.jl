"""
Utilities to add cost functions together
"""

struct DiffEqObjective{F,F2} <: Function
  cost_function::F
  cost_function2::F2
end
(f::DiffEqObjective)(x) = f.cost_function(x)
(f::DiffEqObjective)(x,y) = f.cost_function2(x,y)



function sum_losses(lArray::Array{T,1}, p0) where T<:Function
    Threads.nthreads() == 1 && (@info "Note that restarting julia with multiple threads will increase performance of the generated loss function from sum_losses()")
    # dummy_gs = [convert(SharedArray,deepcopy(p0)) for el in lArray]
    dummy_gs = [deepcopy(p0) for el in lArray]
    dummy_gs2 = [deepcopy(p0) for el in lArray]
    
    # dummy_gs = convert(SharedArray, dummy_gs)
    function cost1(p)
        return reduce(+, [loss(p) for loss in lArray])
    end
    n = length(lArray)
    cs = Vector{Float64}(undef, n)
    # @everywhere lArray2 = lArray
    pure_costs = [(p,g) -> (lArray[i](p,g), g) for i in 1:n]

    function cost2(p,g)
        # @everywhere bar = $fff
        # cs = pmap((i,g) -> lArray[i](p,g), 1:n, dummy_gs)    
        # cs = pmap(1:n, dummy_gs)  do i,g
        #     lArray[i](p,g)
        # end
        pp = [deepcopy(p) for i in 1:n]
        let
            Threads.@threads for ii = 1:n
                cs[ii], dummy_gs[ii] = pure_costs[ii](pp[ii],dummy_gs[ii])  
            end
        end
        c = sum(cs)
        g[:] = reduce(+, dummy_gs)
        return c
    end
    return DiffEqObjective(cost1,cost2)
end
