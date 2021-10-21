"""
MDCProblemJumpStart()
"""
struct MDCProblemJumpStart{A,B,C,D,E,F} <: CurveProblem
    cost::A
    p0::B
    dp0::C
    momentum::D 
    tspan::E
    jumpsize::F
    reinitialise_dp0::Bool
    ## reverse initial direction and signflip curve span if the latter is nonpositive
    function MDCProblemJumpStart(a::A, b::B, c::C, d::D, e::E, f::F, g::Bool) where A where B where C where D where E where F
        if max(e...) <= 0.
            e = map(x -> -x |> abs, e) |> reverse
            c = -c
        end
        new{A,B,C,D,E,F}(a, b, c, d, e, f, g)
    end
end

function MDCProblem(a, b, c, d, e, j::JumpStart)
    MDCProblemJumpStart(a, b, c, d, e, j.jumpsize)
end

function (c::MDCProblemJumpstart)()
    spans = make_spans(c, c.tspan, JumpStart(c.jumpsize))
    cs = map(spans) do span
        mult = sign(span[end])
        return MDCProblemJumpStart(c.cost, c.p0, mult * c.dp0, c.momentum, abs.(span), c.jumpsize)
    end
    spans = map(x -> abs.(x), spans)
    u0s = initial_conditions.(cs)
    u0 = map(span -> initial_conditions(c), spans)
    fs = dynamics.(cs)
    return ODEProblem.(fs, u0s, spans)
end

"""
First need to move p0 in the jumpstart direction. BUT NOT change c.p0.
Then need to make ODEProblem tspan start at the jumpstart time instead of time. ie modify the spans. 
Then need to make the costate: first find an initial u, potentially with reinitialise_dp0==true. Then its easy
Then dynamics are the same
"""
function initial_conditions(c::MDCProblemJumpStart)
    θ₀ = c.p0 + c.jumpsize * (c.dp0 / norm(c.dp0))
    λ₀ = initial_costate(c)
    return cat(θ₀, λ₀, dims=1)
end

function initial_costate(c::MDCProblemJumpStart)
    u = get_initial_velocity(c::MDCProblemJumpStart)
    μ₂ = (-c.momentum + c.cost(c.p0)) / 2.
    λ₀ = -2. * μ₂ * c.dp0 
    return λ₀
end

function get_initial_velocity(c::MDCProblemJumpstart)
    (c.reinitialise_dp0 == false) && (return c.dp0)
    g = param_template(c)
    C = c.cost(p, g)
end