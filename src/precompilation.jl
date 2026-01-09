# Precompilation workload for MinimallyDisruptiveCurves.jl
# This file is included at the end of the main module to trigger precompilation
# of commonly-used code paths during package installation.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Put setup code here (imports, test data setup)
    # This code is NOT precompiled but runs during precompilation
    using LinearAlgebra: norm, dot, eigen
    using ForwardDiff: ForwardDiff
    using OrdinaryDiffEq: Tsit5

    # Simple test cost function
    _p0 = [1.0, 2.0, 3.0]
    function _simple_cost(p)
        a = 1.0
        b = 10.0
        return sum((a .- p) .^ 2) + sum(b .* (p[2:end] .- p[1:(end - 1)] .^ 2) .^ 2)
    end

    function _simple_cost_grad!(p, g)
        n = length(p)
        ε = 1.0e-8
        for i in 1:n
            p_plus = copy(p)
            p_plus[i] += ε
            g[i] = (_simple_cost(p_plus) - _simple_cost(p)) / ε
        end
        return _simple_cost(p)
    end

    @compile_workload begin
        # Put code to precompile here
        # This code runs during precompilation and triggers compilation of the code paths

        # DiffCost creation
        cost = DiffCost(_simple_cost, _simple_cost_grad!)

        # Test cost function calls
        _ = cost(_p0)
        _g = similar(_p0)
        _ = cost(_p0, _g)

        # make_fd_differentiable
        fd_cost = make_fd_differentiable(_simple_cost)
        _ = fd_cost(_p0)

        # TransformationStructure operations
        tr = logabs_transform(_p0)
        tr_cost, newp0 = transform_cost(cost, _p0, tr)
        _ = tr_cost(newp0)

        # MDCProblem creation
        H0 = ForwardDiff.hessian(cost, _p0)
        mom = 100.0
        span = (0.0, 0.5)
        dp0 = (eigen(H0)).vectors[:, 1]
        dp0 = dp0 / norm(dp0)
        eprob = MDCProblem(cost, _p0, dp0, mom, span)

        # Verbose callback setup (precompile callback types)
        cb = [Verbose([CurveDistance([0.25, 0.5])])]

        # evolve - the main entry point
        # Use a very short span to minimize precompilation time
        mdc = evolve(eprob, Tsit5; mdc_callback = cb, verbose = false)

        # MDCSolution methods
        _ = trajectory(mdc)
        _ = costate_trajectory(mdc)
        _ = distances(mdc)
        _ = Δ(mdc)

        # Solution interpolation
        _ = mdc(0.25)
    end
end
