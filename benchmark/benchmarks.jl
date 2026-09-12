using MinimallyDisruptiveCurves, BenchmarkTools
using OrdinaryDiffEqTsit5, ForwardDiff, LinearAlgebra

const SUITE = BenchmarkGroup()

function mass_spring_dynamics!(du, u, p, t)
    m, c, k = p
    position, velocity = u[1], u[2]
    du[1] = velocity
    du[2] = -(c / m) * velocity - (k / m) * position
    return nothing
end

function make_mse_cost(θ_nominal; u0 = [1.0, 0.0], tspan = (0.0, 5.0), dt = 0.2)
    prob_nominal = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_nominal)
    sol_nominal = solve(prob_nominal, Tsit5(); saveat = dt)
    target_times = sol_nominal.t
    target_positions = [sol[1] for sol in sol_nominal.u]

    f = function (θ)
        if any(θ .<= 1.0e-3)
            return 100.0 + sum(abs2, min.(zero(eltype(θ)), θ))
        end
        prob = ODEProblem(mass_spring_dynamics!, u0, tspan, θ)
        sol = solve(prob, Tsit5(); saveat = target_times)
        current_positions = [s[1] for s in sol.u]
        return sum(abs2, current_positions .- target_positions) / length(target_times)
    end

    grad! = (g, θ) -> ForwardDiff.gradient!(g, f, θ)
    return CostFunction(f, grad!)
end

θ_nominal = [1.0, 0.5, 5.0]
core_cost = make_mse_cost(θ_nominal)
transformed_cost = TransformedCost(core_cost, TransformChain())

# =============================================================================
# Cost evaluation + gradient
# =============================================================================

SUITE["cost"] = BenchmarkGroup()

SUITE["cost"]["evaluate"] = @benchmarkable $core_cost.f($θ_nominal)
SUITE["cost"]["gradient"] = @benchmarkable begin
    g = zeros(3)
    $core_cost.grad!(g, $θ_nominal)
end

# =============================================================================
# MDCProblem construction and curve integration
# =============================================================================

SUITE["mdc"] = BenchmarkGroup()

SUITE["mdc"]["problem"] = @benchmarkable MDCProblem(
    $transformed_cost, $θ_nominal, $θ_nominal, 1.0;
    names = [:mass, :damping, :stiffness]
)

sys = MDCProblem(
    transformed_cost, θ_nominal, θ_nominal, 1.0;
    names = [:mass, :damping, :stiffness]
)

SUITE["mdc"]["solve"] = @benchmarkable MDCSolve(
    $sys; span = MDCSpan(-0.5, 0.5), alg = Tsit5()
) seconds = 180

stabilizer = mdc_momentum_readjustment(sys; tol = 1.0e-3)

SUITE["mdc"]["solve_stabilized"] = @benchmarkable MDCSolve(
    $sys; span = MDCSpan(-0.5, 0.5), alg = Tsit5(),
    callback = CallbackSet($stabilizer)
) seconds = 180
