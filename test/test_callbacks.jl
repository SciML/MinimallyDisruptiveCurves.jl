using SafeTestsets

@safetestset "Safety Controls & System Guards Unit Tests" begin
    using MinimallyDisruptiveCurves
    using Test
    using OrdinaryDiffEq
    using ForwardDiff

    # Re-use your exact mass-spring physics engine
    function mass_spring_dynamics!(du, u, p, t)
        m, c, k = p
        position, velocity = u[1], u[2]
        du[1] = velocity
        du[2] = -(c / m) * velocity - (k / m) * position
        return nothing
    end

    function make_callback_cost(θ_nominal; u0=[1.0, 0.0], tspan=(0.0, 5.0), dt=0.2)
        prob_nominal = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_nominal)
        sol_nominal = solve(prob_nominal, Tsit5(), saveat=dt)
        target_times = sol_nominal.t
        target_positions = [sol[1] for sol in sol_nominal.u]
        
        f = function(θ)
            if any(θ .<= 1e-3)
                return 100.0 + sum(abs2, min.(zero(eltype(θ)), θ))
            end
            prob = ODEProblem(mass_spring_dynamics!, u0, tspan, θ)
            sol = solve(prob, Tsit5(), saveat=target_times)
            current_positions = [s[1] for s in sol.u]
            return sum(abs2, current_positions .- target_positions) / length(target_times)
        end
        
        grad! = (g, θ) -> ForwardDiff.gradient!(g, f, θ)
        return CostFunction(f, grad!)
    end

    @testset "Mass-Spring Boundary Guard Violations" begin
        # Start with a valid physical system
        θ_nominal = [1.0, 0.5, 5.0]  # m=1.0, c=0.5, k=5.0
        
        # CRITICAL: Point the velocity vector straight down towards 0 mass!
        # This guarantees that the forward integration path will try to destroy the system.
        dθ_crash  = [-1.0, 0.0, 0.0] 
        H = 10.0 # Give it plenty of momentum capacity to run

        core_cost = make_callback_cost(θ_nominal)
        sys = MDCSystem(TransformedCost(core_cost, TransformChain()), θ_nominal, dθ_crash, H; names=[:mass, :damping, :stiffness])
        
        safety_cb = mdc_safety_callback(sys)

        # Run a symmetric span. Both directions initialize cleanly because θ_nominal is valid.
        mdc_curves = MDCSolve(sys, span=MDCSpan(-20.0, 20.0), callback=safety_cb)

        # Verify that the trajectory moving towards zero mass was caught and stopped
        @test mdc_curves.positive_sol !== nothing
        
        # The callback should step in, print its warning, and terminate the solver gracefully
        @test mdc_curves.positive_sol.retcode == ReturnCode.Terminated
        
        # Confirm that the guard stopped it BEFORE mass could become zero or negative
        final_mass = mdc_curves.positive_sol.u[end][1]
        @test final_mass > 0.0
    end
end
