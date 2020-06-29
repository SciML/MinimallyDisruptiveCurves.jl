
abstract type AbstractCurve end

mutable struct testt 
    x::Union{Nothing, Int64}
end


mutable struct MinimallyDisruptiveCurve{S, F} <: AbstractCurve
    sol::S
    N::Int64
    costf::F
end



function MinimallyDisruptiveCurve(sol, costf=nothing)
    N = length(sol.prob.u0) ÷ 2
    return MinimallyDisruptiveCurve(sol, N, costf)    
end


function (mdc::MinimallyDisruptiveCurve)(t::N) where N <: Number
    return (states = (@view mdc.sol(t)[1:mdc.N]), costates = (@view mdc.sol(t)[mdc.N+1:end]))
end

function (mdc::MinimallyDisruptiveCurve)(ts::A) where A <: Array
    states_ = Array{Float64,2}(undef, mdc.N, length(ts))
    costates_ = Array{Float64,2}(undef, mdc.N, length(ts))
    for (i,el) in enumerate(ts)
        states_[:,i] = mdc(el)[:states]
        costates_[:,i] = mdc(el)[:costates]
    end
    return (states = states_, costates = costates_)
end


function add_cost(mdc::MinimallyDisruptiveCurve, costf)
    return MinimallyDisruptiveCurve(mdc.sol, mdc.N, costf)
end

trajectory(mdc::MinimallyDisruptiveCurve) = Array(mdc.sol)[1:mdc.N, :]
trajectory(mdc::MinimallyDisruptiveCurve, ts) = mdc(ts)[:states]

costate_trajectory(mdc::MinimallyDisruptiveCurve) = Array(mdc.sol)[mdc.N+1:end,:]
costate_trajectory(mdc::MinimallyDisruptiveCurve, ts) = mdc(ts)[:costates]

distances(mdc::MinimallyDisruptiveCurve) = mdc.sol.t
Δ(mdc::MinimallyDisruptiveCurve) = trajectory(mdc) .- mdc(0.)[:states]
Δ(mdc::MinimallyDisruptiveCurve, ts) = mdc(ts)[:states] .- mdc(0.)[:states]


function cost_trajectory(mdc::MinimallyDisruptiveCurve, ts)
    if mdc.costf === nothing
        @warn "MinimallyDisruptiveCurve struct has no cost function. You need to run mdc = add_cost(mdc, costf)"
        return
    else
        return [mdc.costf(el) for el in eachcol(mdc(ts)[:states])]
    end
end


@recipe function f(mdc::MinimallyDisruptiveCurve; pnames = nothing, idxs=nothing, what = :trajectory)
    if idxs === nothing
        idxs = biggest_movers(mdc, 5)
    end
    # if !(names === nothing)
    #     labels --> names[idxs]
    # end
    #["hi" "lo" "lo" "hi" "lo"]


    tfirst = mdc.sol.t[1]
    tend = mdc.sol.t[end]

    layout := (1,1)  
    bottom_margin := :match

    if what == :trajectory
        @series begin
            if !(pnames === nothing)
                label -->  reshape(pnames[idxs], 1, :)
            end
            title --> "change in parameters over minimally disruptive curve"
            xguide --> "distance"
            yguide --> "change in parameters"
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
                Δ(mdc, tfirst)[idxs]
            end
        end
    end


 
end