
abstract type AbstractCurveSpec end
abstract type AbstractCurve end

mutable struct curveProblem{F,P,D,M,S} <: AbstractCurveSpec
    cost::F
    p0::P
    dp0::D
    momentum::M
    tspan::S
end

specify_curve(cost, p0, dp0, momentum, tspan) = curveProblem(cost, p0, dp0, momentum, tspan)

specify_curve(;cost=nothing, p0=nothing, dp0=nothing,momentum=nothing,tspan=nothing) = curveProblem(cost,p0,dp0,momentum,tspan)


function evolveODE(du ,u , p, t, cost, ∇C, N, H, θ₀)
 
    θ = u[1:N] # current parameter vector
    λ = u[N+1:end] #current costate vector
    dist = sum((θ - θ₀).^2) # should = t actually? check and replace?
    C = cost(θ, ∇C) #also updates ∇C as a mutable
    μ2 = (C-H)/2
    μ1 = dist > 1e-3 ?  (λ'*λ - 4*μ2^2 )/(λ'*(θ - θ₀)) : 0 
        # if mu1 < -1e-4 warn of numerical issue
        # if mu1 > 1e-3 and dist > 1e-3 then set mu1 = 0
    du[1:N] = @. (-λ + μ1*(θ - θ₀))/(2*μ2) # ie dθ
    du[1:N] /= (sqrt(sum((du[1:N]).^2)))
    damping_constant = (λ'*du[1:N])/(H-C)  #theoretically = 1 but not numerically
    du[N+1:end] = @. (μ1*du[1:N] - ∇C)*damping_constant # ie dλ
    res = λ  + 2*μ2*du[1:N]
    return nothing
end


function make_ODEProblem(c::C) where C <: curveProblem
    N = length(c.p0)
    ∇C = copy(c.dp0)
    λ0 = initial_costate(c.dp0, c.momentum, c.cost(c.p0))
    u0 = cat(c.p0, λ0, dims=1)
    f = (du,u,p,t) -> evolveODE(du,u,p,t, c.cost, ∇C, N, c.momentum, c.p0)
    return ODEProblem(f, u0, c.tspan)
end


mutable struct MinimallyDisruptiveCurve{S, F} <: AbstractCurve
    sol::S
    N::Int64
    cost::F
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
    if mdc.cost === nothing
        @warn "MinimallyDisruptiveCurve struct has no cost function. You need to run mdc = add_cost(mdc, cost)"
        return
    else
        return [mdc.cost(el) for el in eachcol(mdc(ts)[:states])]
    end
end


@recipe function f(mdc::MinimallyDisruptiveCurve; pnames = nothing, idxs=nothing, what = :trajectory)
    if idxs === nothing
        num = min(5, mdc.N)
        idxs = biggest_movers(mdc, num)
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