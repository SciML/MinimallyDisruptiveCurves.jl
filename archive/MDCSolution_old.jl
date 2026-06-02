abstract type AbstractCurveSolution end

"""
    mutable struct MDCSolution{S, F} <: AbstractCurve

holds solution for minimally disruptive curve.
fields: sol, N, cost
sol is a conventional ODESolution structure
"""
mutable struct MDCSolution{S, F} <: AbstractCurveSolution
    sol::S
    N::Int64
    cost::F
end

"""
    function MDCSolution(sol, costf=nothing)

returns an mdc::MDCSolution out of a sol::ODESolution
"""
function MDCSolution(sol, costf = nothing)
    N = length(sol.prob.u0) ÷ 2
    return MDCSolution(sol, N, costf)
end

"""
    function (mdc::MDCSolution)(t::N) where N <: Number

returns array view of states and costates of curve at distance t
"""
function (mdc::MDCSolution)(t::N) where {N <: Number}
    return (
        states = (@view mdc.sol(t)[1:mdc.N]),
        costates = (@view mdc.sol(t)[(mdc.N + 1):end])
    )
end

"""
    function (mdc::MDCSolution)(ts::A) where A <: Array

returns array of states and costates of curve at distances t
"""
function (mdc::MDCSolution)(ts::A) where {A <: Array}
    states_ = Array{Float64, 2}(undef, mdc.N, length(ts))
    costates_ = Array{Float64, 2}(undef, mdc.N, length(ts))
    for (i, el) in enumerate(ts)
        states_[:, i] = mdc(el)[:states]
        costates_[:, i] = mdc(el)[:costates]
    end
    return (states = states_, costates = costates_)
end

trajectory(mdc::MDCSolution) = Array(mdc.sol)[1:mdc.N, :]
trajectory(mdc::MDCSolution, ts) = mdc(ts)[:states]

costate_trajectory(mdc::MDCSolution) = Array(mdc.sol)[(mdc.N + 1):end, :]
costate_trajectory(mdc::MDCSolution, ts) = mdc(ts)[:costates]

distances(mdc::MDCSolution) = mdc.sol.t
Δ(mdc::MDCSolution) = trajectory(mdc) .- mdc(0.0)[:states]
Δ(mdc::MDCSolution, ts) = mdc(ts)[:states] .- mdc(0.0)[:states]

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

function Base.show(io::IO, m::MIME"text/plain", M::MDCSolution)
    return show(io, m, M.sol)
end
