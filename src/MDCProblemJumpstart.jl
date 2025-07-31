"""
    MDCProblemJumpStart(cost, p0 dp0, momentum, tspan, jumpsize, reinitialise_dp0)

Do I want to add an option: (reinitialise_dp0 == true) && give option to recalculate hessian
"""
struct MDCProblemJumpStart{A, B, C, D, E, F} <: CurveProblem
    cost::A
    p0::B
    dp0::C
    momentum::D
    tspan::E
    jumpsize::F
    reinitialise_dp0::Bool
    ## reverse initial direction and signflip curve span if the latter is nonpositive
    function MDCProblemJumpStart(a::A, b::B, c::C, d::D, e::E, f::F,
            g::Bool) where {A} where {B} where {C} where {D} where {E} where {F}
        if max(e...) <= 0.0
            e = map(x -> -x |> abs, e) |> reverse
            c = -c
        end
        new{A, B, C, D, E, F}(a, b, c, d, e, f, g)
    end
end
isjumped(c::MDCProblemJumpStart) = JumpStart(c.jumpsize)
whatdynamics(c::MDCProblemJumpStart) = MDCDynamics()

function jumpstart(m::MDCProblem, jumpsize, reinitialise_dp0 = false)
    return MDCProblemJumpStart(
        m.cost, m.p0, m.dp0, m.momentum, m.tspan, jumpsize, reinitialise_dp0)
end

function (j::JumpStart)(m::MDCProblem; reinitialise_dp0 = false)
    return MDCProblemJumpStart(
        m.cost, m.p0, m.dp0, m.momentum, m.tspan, j.jumpsize, reinitialise_dp0)
end

function (c::MDCProblemJumpStart)()
    spans = make_spans(c, c.tspan, ZeroStart())
    cs = map(spans) do span
        mult = sign(span[end])
        return MDCProblemJumpStart(c.cost, c.p0, mult * c.dp0, c.momentum,
            abs.(span), c.jumpsize, c.reinitialise_dp0)
    end
    spans = map(x -> abs.(x), spans)
    u0s = initial_conditions.(cs)
    fs = dynamics.(cs)
    return ODEProblem.(fs, u0s, spans)
end

function make_spans(c::CurveProblem, span, j::JumpStart)
    spans = make_spans(c, span, ZeroStart())
    spans = map(spans) do span
        (span[1] + 0.1 * sign(span[2]), span[2])
    end
    return spans
end

"""
First need to move p0 in the jumpstart direction. BUT NOT change c.p0.
Then need to make ODEProblem tspan start at the jumpstart time instead of time. ie modify the spans.
Then need to make the costate: first find an initial u, potentially with reinitialise_dp0==true. Then its easy
Then dynamics are the same
"""
function initial_conditions(c::MDCProblemJumpStart)
    θ₀ = c.p0 + get_jump(c)
    λ₀ = initial_costate(c)
    return cat(θ₀, λ₀, dims = 1)
end

function get_jump(c::CurveProblem)
    get_jump(c, isjumped(c))
end
get_jump(c, ::ZeroStart) = zero.(param_template(c))
function get_jump(c, j::JumpStart)
    return j.jumpsize * (c.dp0 / norm(c.dp0))
end

function initial_costate(c::MDCProblemJumpStart)
    u = get_initial_velocity(c::MDCProblemJumpStart)
    μ₂ = (-c.momentum + c.cost(c.p0)) / 2.0
    λ₀ = -2.0 * μ₂ * u  # + μ₁*get_jump(c), but μ₁ = 0 by complementary slackness at this point
    return λ₀
end

"""
Algorithm:
Let y = θ - θ₀;
f(x) = ∇Cᵀx; at θ
g₁(x) = xᵀx - 1;
g₂(x) = - xᵀy
Then optimisation problem is:
minₓ f(x) subject to
gᵢ(x) ≤ 0.
In other words, find a direction that maximally anticorrelates with the gradient, but has norm ≤ 1 and is pointing away from the curve origin. Norm = 1 would be ideal, but deconvexifies.
KKT conditions give:
∇C + 2μ₁x - μ₂y = 0;
μᵢ ⩾ 0 + complementary slackness
Analytic solution:
Case 1: ∇Cᵀy ≤ 0. Then μ₂ = 0 from complementary slackness and x ∝ - ∇C
Case 2: ∇Cᵀ ≥ 0. then
2μ₁x = μ₂y - ∇C
⇒   μ₂ = ∇Cᵀy / yᵀy
⇒   ∇Cᵀx ≥ 0 → x = 0
∇Cᵀx ≤ 0 → μ₁ s.t. norm(x) = 1
Issues:

  - ∇C might be quite noisy close to the minimum. Might want the option of a second order condition involving hessian recalculation for cheaper problems. IE
  - What do we do if x = 0? For now, keep the old dp0. Because at least that is in a shallow direction of the old Hessian. In future, could provide option to recalculate Hessian OR reuse Hessian at p0, which would be a good estimate of the new Hessian. If we have the Hessian, the optimisation problem would change:
    f(x) = ∇Cᵀx + 0.5 xᵀ∇²Cx ; at θ
    KKT:
    ∇C + ∇²Cx + 2μ₁x - μ₂y = 0
    (∇²C + 2μ₁𝕀)x = μ₂y - ∇C
    And then go through the μᵢ = or ≂̸ 0 cases.
"""
function get_initial_velocity(c::MDCProblemJumpStart)
    (c.reinitialise_dp0 == false) && (return c.dp0)
    ∇C = param_template(c)
    y = get_jump(c)
    new_p0 = c.p0 + y
    c.cost(new_p0, ∇C)
    d = dot(∇C, y)
    if d ≤ 0
        new_dp0 = -∇C/(norm(∇C))
    else
        μ₂ = dot(∇C, y) / sum(abs2, y)
        x = μ₂*y - ∇C;
        nx = norm(x)
        new_dp0 = x/nx
    end
    if c.cost(new_p0 + 0.001new_dp0) > c.cost(new_p0 + 0.001c.dp0)
        @info("couldn't cheaply find a better initial curve direction dp0 for the jumpstarted problem. will not reinitialise dp0. Provide your own if you want")
        return c.dp0
    else
        @info("cheaply found a better initial curve direction dp0 for the jumpstarted problem. . re-initialising dp0")
        return x / nx
    end
end
