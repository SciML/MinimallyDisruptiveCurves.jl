using SafeTestsets

@safetestset "Solver Pipelines Integration Tests" begin
    using MinimallyDisruptiveCurves
    using Test
    using LinearAlgebra
    using OrdinaryDiffEq
    using ForwardDiff

    # ====================================================================
    # --- 1. 2D Mass-Spring-Damper System Definition ---
    # ====================================================================
    function mass_spring_dynamics!(du, u, p, t)
        m, c, k = p
        position, velocity = u[1], u[2]
        du[1] = velocity
        du[2] = -(c / m) * velocity - (k / m) * position
        return nothing
    end

    # Minimal dynamic MSE cost function factory using ForwardDiff
    function make_mse_cost_function(θ_nominal; u0 = [1.0, 0.0], tspan = (0.0, 5.0), dt = 0.2)
        prob_nominal = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_nominal)
        sol_nominal = solve(prob_nominal, Tsit5(), saveat = dt)
        target_times = sol_nominal.t
        target_positions = [sol[1] for sol in sol_nominal.u]

        f = function (θ)
            if any(θ .<= 1.0e-3)
                return 100.0 + sum(abs2, min.(zero(eltype(θ)), θ))
            end
            prob = ODEProblem(mass_spring_dynamics!, u0, tspan, θ)
            sol = solve(prob, Tsit5(), saveat = target_times)
            current_positions = [s[1] for s in sol.u]
            return sum(abs2, current_positions .- target_positions) / length(target_times)
        end

        grad! = (g, θ) -> ForwardDiff.gradient!(g, f, θ)
        return CostFunction(f, grad!)
    end

    # ====================================================================
    # --- 2. End-to-End Solver & Invariant Tests ---
    # ====================================================================
    @testset "Mass-Spring-Damper Invariant Preservation" begin
        # Setup nominal points and initial conditions
        θ_nominal = [1.0, 0.5, 5.0]
        u0_physical = [1.0, 0.0]
        tspan_physical = (0.0, 5.0)

        core_cost = make_mse_cost_function(θ_nominal, u0 = u0_physical, tspan = tspan_physical)
        transformed_cost = TransformedCost(core_cost, TransformChain())

        # Initialize tracking along the null-space trajectory direction
        sys = MDCProblem(transformed_cost, θ_nominal, θ_nominal, 1.0; names = [:mass, :damping, :stiffness])
        stabilizer = mdc_momentum_readjustment(sys; tol = 1.0e-3)

        # Run integration
        mdc_curves = MDCSolve(sys, span = MDCSpan(-3.0, 3.0), callback = CallbackSet(stabilizer))

        # Ensure both paths generated tracking solutions successfully
        @test mdc_curves.positive_sol !== nothing
        @test mdc_curves.negative_sol !== nothing

        # Validate final state ratios match initial system configurations precisely
        final_state = mdc_curves.negative_sol.u[end]
        θ_explored = final_state[1:3]

        initial_cm = θ_nominal[2] / θ_nominal[1]
        initial_km = θ_nominal[3] / θ_nominal[1]
        explored_cm = θ_explored[2] / θ_explored[1]
        explored_km = θ_explored[3] / θ_explored[1]

        # Invariants must match closely along the minimally disruptive direction
        @test explored_cm ≈ initial_cm rtol = 1.0e-2
        @test explored_km ≈ initial_km rtol = 1.0e-2

        # Final MSE structural variance should be effectively zero (or very close to it)
        @test core_cost.f(θ_explored) < 1.0e-4
    end
end
