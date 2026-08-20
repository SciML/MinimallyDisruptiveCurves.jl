using ForwardDiff

# ====================================================================
# --- Mass Spring ---
# ====================================================================

# Standard 2D mass-spring-damper vector field: ẋ = v, v̇ = -(c/m)v - (k/m)x
function mass_spring_dynamics!(du, u, p, t)
    m, c, k = p
    position, velocity = u[1], u[2]

    du[1] = velocity
    du[2] = -(c / m) * velocity - (k / m) * position
    return nothing
end


"""
    make_mse_cost_function(θ_nominal; u0=[1.0, 0.0], tspan=(0.0, 10.0), dt=0.1)

Generates a clean `CostFunction` instance. It pre-computes a reference trajectory 
using `θ_nominal` and evaluates the MSE deviation for any test parameter vector θ.
"""
function make_mass_spring_mse_cost_function(θ_nominal; u0 = [1.0, 0.0], tspan = (0.0, 10.0), dt = 0.1)
    # 1. Generate the immutable reference trajectory data
    prob_nominal = ODEProblem(mass_spring_dynamics!, u0, tspan, θ_nominal)
    sol_nominal = solve(prob_nominal, Tsit5(), saveat = dt)

    target_times = sol_nominal.t
    target_positions = [sol[1] for sol in sol_nominal.u]

    # 2. Define the objective function closure (f)
    # NOTE: We remove the manual Float64 type-restrictions so Dual numbers can pass through
    function f(θ)
        if any(θ .<= 1.0e-3)
            # Ensure the penalty return type matches the input dual/real element type dynamically
            return 100.0 + sum(abs2, min.(zero(eltype(θ)), θ))
        end

        # Pass θ directly—OrdinaryDiffEq automatically handles dual-number parameters!
        prob = ODEProblem(mass_spring_dynamics!, u0, tspan, θ)
        sol = solve(prob, Tsit5(), saveat = target_times)

        current_positions = [s[1] for s in sol.u]
        return sum(abs2, current_positions .- target_positions) / length(target_times)
    end

    # 3. Define the exact Automatic Differentiation gradient closure (grad!)
    function grad!(g, θ)
        ForwardDiff.gradient!(g, f, θ)
        return g
    end

    return CostFunction(f, grad!)
end

# ===================================================================
# --- Lotka-Volterra ---
# ===================================================================

function lotka_volterra_dynamics!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
    return nothing
end

function extract_statistics(p, u₀, tspan, dt)
    lv_problem = ODEProblem(lotka_volterra_dynamics!, u₀, tspan, p)
    lv_sol = solve(lv_problem, Tsit5(), saveat = dt)

    mean_prey = sum([sol[1] for sol in lv_sol.u]) / length(lv_sol.u)
    max_predator = maximum([sol[2] for sol in lv_sol.u])

    return [mean_prey, max_predator]
end

function make_lv_cost_function(p_nom, u₀, tspan, dt = 0.1)
    measured_stats = extract_statistics(p_nom, u₀, tspan, dt)
    function cost(p)
        stats = extract_statistics(p, u₀, tspan, dt)
        return sum(abs2, stats .- measured_stats)
    end
    function grad!(g, p)
        ForwardDiff.gradient!(g, cost, p)
        return nothing
    end
    return CostFunction(cost, grad!)
end

# ===================================================================
# --- NFKB ---
# ===================================================================

function nfkb_loss_function(x, p_tuple)
    # Destructure context tuple
    odeprob, ts, truth, setter, diffcache, obs_symbols = p_tuple

    ps = parameter_values(odeprob)
    buffer = get_tmp(diffcache, x)

    # Block-copy baseline values (the non-tunables stay untouched elsewhere in `ps`)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])

    # Type-safe structural parameter container replacement for ForwardDiff
    ps_updated = replace(Tunable(), ps, buffer)

    # Mutate only our active dual/float optimization array
    setter(ps_updated, x)

    # Fast inferred problem recreation
    newprob = remake(odeprob; p = ps_updated)
    sol = solve(newprob, Tsit5(); saveat = ts)

    if sol.retcode != SciMLBase.ReturnCode.Success
        return eltype(x)(Inf) # Strict type stability for dual-number propagation
    end

    # Extract states cleanly via targeted tracking symbols
    current_data = sol(ts, idxs = obs_symbols)

    # Allocation-free MSE over the exact matrix of specified states
    return sum(abs2, truth .- current_data) / length(truth)
end

function make_sin_cost_function(A, ω)
    function cost(p)
        sin_value = A * sin(ω * p[1])
        return (p[2] - sin_value)^2.0
    end
    function grad!(g, p)
        x, y = p[1], p[2]
        dy = 2.0 * (y - A * sin(ω * x))
        g[1] = -dy * A * ω * cos(ω * x)
        g[2] = dy
        return g
    end
    return CostFunction(cost, grad!)
end