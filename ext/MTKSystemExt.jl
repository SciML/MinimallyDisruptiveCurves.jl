module MTKSystemExt

using MinimallyDisruptiveCurves
using ModelingToolkit
using OrdinaryDiffEq
import ModelingToolkit: parameters, ModelingToolkitBase
using SciMLStructures: Tunable, canonicalize, replace 
using DifferentiationInterface 
const DI = DifferentiationInterface

"""
    mtk_parameter_mapping(mtk_sys::ODESystem, alg = Tsit5(); tspan = (0.0, 30000.0), solve_kwargs...)

Analyze a compiled `ModelingToolkit.ODESystem` and extract a structured parameter interface 
compatible with optimization and sensitivity sweeps.

# Arguments
- `mtk_sys::ODESystem`: A finalized, completed ModelingToolkit ODE system model.
- `alg`: An OrdinaryDiffEq solver algorithm (defaults to `Tsit5()`).

# Keyword Arguments
- `tspan::Tuple{Float64, Float64}`: The time interval for the differential equation simulation.
- `solve_kwargs...`: Optional configuration arguments passed natively down to `OrdinaryDiffEq.solve`.

# Returns
A `NamedTuple` with the following fields:
- `simulator::Function`: A zero-allocation closure `f(θ)` that accepts a raw vector of new 
  parameter values, maps them back into the compiled `MTKParameters` structural layout tree, 
  and returns an `ODESolution`.
- `θ_nominal::Vector{Float64}`: The default master parameter values extracted dynamically from 
  the system's compiled layout buffer.
- `names::Vector{Symbol}`: The parameter names matching the elements and indices of `θ_nominal`.

# Example
```julia
    mapping = mtk_parameter_mapping(sys) 
    sol = mapping.simulator(mapping.θ_nominal)
```
"""
function MinimallyDisruptiveCurves.mtk_parameter_mapping(
    mtk_sys::ODESystem, 
    alg = OrdinaryDiffEq.Tsit5(); 
    tspan = (0.0, 30000.0), 
    solve_kwargs...
)
    base_prob = ODEProblem(mtk_sys, Dict(), tspan)
    ps_structure = base_prob.p
    initial_guess, repack, alias = canonicalize(Tunable(), ps_structure)
    
    tunable_symbols = Symbol[]
    if hasproperty(ps_structure, :tunables)
        for key in keys(ps_structure.tunables)
            push!(tunable_symbols, Symbol(key))
        end
    else
        structural_vars = ModelingToolkit.parameters(mtk_sys)
        scalarized_vars = Symbolics.scalarize(structural_vars)
        for var in scalarized_vars
            if ModelingToolkitBase.parameter_index(mtk_sys, var) !== nothing
                push!(tunable_symbols, Symbol(var))
            end
        end
    end
    
    if length(initial_guess) != length(tunable_symbols)
        tunable_symbols = [Symbol("p_$i") for i in 1:length(initial_guess)]
    end
    
    simulator = function (θ)
        ps_updated = replace(Tunable(), ps_structure, θ)
        local_prob = remake(base_prob; p = ps_updated)
        return OrdinaryDiffEq.solve(local_prob, alg; solve_kwargs...)
    end
    
    return (simulator = simulator, θ_nominal = initial_guess, names = tunable_symbols)
end


"""
    mtk_cost_mapping(mtk_sys::ODESystem, user_cost::Function, backend::AbstractADType, alg = Tsit5(); tspan = (0.0, 30000.0), solve_kwargs...)

Wrap a compiled `ModelingToolkit.ODESystem` and a user-defined trajectory evaluation function 
into a unified, backend-agnostic gradient cost interface.

# Arguments
- `mtk_sys::ODESystem`: A completed ModelingToolkit ODE system model.
- `user_cost::Function`: A user function `f(sol::ODESolution)` that processes a trajectory 
  and returns a scalar cost value (e.g., Mean Squared Error).
- `backend::AbstractADType`: A valid DifferentiationInterface AD backend object (e.g., `AutoForwardDiff()`).
- `alg`: An OrdinaryDiffEq solver algorithm (defaults to `Tsit5()`).

# Keyword Arguments
- `tspan::Tuple{Float64, Float64}`: The simulation time interval.
- `solve_kwargs...`: Optional configurations passed down to the underlying ODE solver.

# Returns
A `NamedTuple` with the following fields:
- `cost_function::CostFunction`: A container holding the scalar cost evaluation function `f(z)` 
  and its corresponding in-place gradient calculator `grad!(gz, z)` optimized with the chosen 
  AD backend preparation tools.
- `θ_nominal::Vector{Float64}`: The vector of default parameter values.
- `names::Vector{Symbol}`: The symbolic labels aligned 1-to-1 with the parameter vector.

# Example
```julia
using DifferentiationInterface

cf_result = mtk_cost_mapping(sys, my_mse_cost, AutoForwardDiff())
physical_cost = cf_result.cost_function

# Evaluate cost and trace parameter gradients simultaneously
gz = similar(cf_result.θ_nominal)
val = physical_cost.f(cf_result.θ_nominal)
physical_cost.grad!(gz, cf_result.θ_nominal)
```
"""
function MinimallyDisruptiveCurves.mtk_cost_mapping(
    mtk_sys::ODESystem,
    user_cost::Function,
    backend::DI.ADTypes.AbstractADType, 
    alg = OrdinaryDiffEq.Tsit5();
    tspan = (0.0, 30000.0),
    solve_kwargs...
)
    mapping = MinimallyDisruptiveCurves.mtk_parameter_mapping(
        mtk_sys, alg; tspan=tspan, solve_kwargs...
    )
    sim = mapping.simulator

    physical_cost_f = function(z)
        sol = sim(z)
        return user_cost(sol)
    end

    extras = DI.prepare_gradient(physical_cost_f, backend, mapping.θ_nominal; strict=Val(false))

    physical_grad! = function(gz, z)
        DI.gradient!(physical_cost_f, gz,extras, backend, z)
        return gz
    end

    cf = MinimallyDisruptiveCurves.CostFunction(physical_cost_f, physical_grad!)

    return (cost_function = cf, θ_nominal = mapping.θ_nominal, names = mapping.names)
end

end # module
