abstract type AbstractCurveSolution end


"""
    mutable struct MDCSolution{S, F} <: AbstractCurve
holds solution for minimally disruptive curve. 
fields: sol, N, cost
sol is a conventional ODESolution structure
"""
mutable struct MDCSolution{S,F} <: AbstractCurveSolution
    sol::S
    N::Int64
    cost::F
end


"""
    function MDCSolution(sol, costf=nothing)
returns an mdc::MDCSolution out of a sol::ODESolution
"""
function MDCSolution(sol, costf=nothing)
    N = length(sol.prob.u0) ÷ 2
    return MDCSolution(sol, N, costf)    
end

"""
    function (mdc::MDCSolution)(t::N) where N <: Number
returns array view of states and costates of curve at distance t
"""
function (mdc::MDCSolution)(t::N) where N <: Number
    return (states = (@view mdc.sol(t)[1:mdc.N]), costates = (@view mdc.sol(t)[mdc.N + 1:end]))
end

"""
    function (mdc::MDCSolution)(ts::A) where A <: Array
returns array of states and costates of curve at distances t
"""
function (mdc::MDCSolution)(ts::A) where A <: Array
    states_ = Array{Float64,2}(undef, mdc.N, length(ts))
    costates_ = Array{Float64,2}(undef, mdc.N, length(ts))
    for (i, el) in enumerate(ts)
        states_[:,i] = mdc(el)[:states]
        costates_[:,i] = mdc(el)[:costates]
    end
    return (states = states_, costates = costates_)
end


trajectory(mdc::MDCSolution) = Array(mdc.sol)[1:mdc.N, :]
trajectory(mdc::MDCSolution, ts) = mdc(ts)[:states]

costate_trajectory(mdc::MDCSolution) = Array(mdc.sol)[mdc.N + 1:end,:]
costate_trajectory(mdc::MDCSolution, ts) = mdc(ts)[:costates]

distances(mdc::MDCSolution) = mdc.sol.t
Δ(mdc::MDCSolution) = trajectory(mdc) .- mdc(0.)[:states]
Δ(mdc::MDCSolution, ts) = mdc(ts)[:states] .- mdc(0.)[:states]

"""
    cost_trajectory(mdc::MDCSolution, ts)
calculates cost on mdc curve at each point in the array/range ts
"""
function cost_trajectory(mdc::MDCSolution, ts)
    if mdc.cost === nothing
        @warn "MDCSolution struct has no cost function. You need to run mdc = add_cost(mdc, cost)"
        return
    else
        return [mdc.cost(el) for el in eachcol(mdc(ts)[:states])]
    end
end

"""
plot recipe for ::MDCSolution
kwargs: pnames are array of parameter names
idxs: are parameter indices to plot
what ∈ (:trajectory, :final_changes) determines the plot type
"""
@recipe function f(mdc::MDCSolution; pnames=nothing, idxs=nothing, what=:trajectory)
    if idxs === nothing
        num = min(5, mdc.N)
        idxs = biggest_movers(mdc, num)
    end
    # if !(names === nothing)
    #     labels --> names[idxs]
    # end
    # ["hi" "lo" "lo" "hi" "lo"]


    tfirst = mdc.sol.t[1]
    tend = mdc.sol.t[end]

    layout := (1, 1)  
    bottom_margin := :match

    if what == :trajectory
        @series begin
            if !(pnames === nothing)
                label -->  reshape(pnames[idxs], 1, :)
            end
            title --> "change in parameters over minimally disruptive curve"
            xguide --> "distance"
            yguide --> "Δ parameters"
            distances(mdc), Δ(mdc)[idxs,:]'
        end
    end

    if what == :final_changes
        @series begin
            title --> "biggest changers"
        seriestype := :bar
            label --> "t=$tend"
            xticks --> (1:5, reshape(pnames[idxs], 1, :))
            xrotation --> 90
            Δ(mdc, tend)[idxs] 
        end
        if tfirst < 0.
            @series begin
                label --> "t=$tfirst"
                seriestype := :bar
                xticks --> (1:5, reshape(pnames[idxs], 1, :))
                xrotation --> 90
                Δ(mdc, tfirst)[idxs]
            end
        end
    end
end