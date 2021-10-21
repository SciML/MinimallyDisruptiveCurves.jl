abstract type CurveInfoSnippet end
struct EmptyInfo <: CurveInfoSnippet end

"""
    CurveDistance(timepoints::AbstractRange)
    CurveDistance(timepoints::Vector{<:AbstractFloat})
Print @info REPL output when curve distance reaches each of the timepoints::Vector{<:AbstractFloat}

Example
c = CurveDistance(0.1:0.1:2.1)

c can then be used as an argument to Verbose(), which then goes as a keyword argument into evolve.
"""
struct CurveDistance{V <: AbstractFloat} <: CurveInfoSnippet 
    timepoints::Vector{V}
end

"""
    HamiltonianResidual(timepoints::AbstractRange)
    HamiltonianResidual(timepoints::Vector{<:AbstractFloat})
Print @info REPL output on the numerical value of dHdu, the u-derivative of the Hamiltonian, at timepoints::Vector{<:AbstractFloat}

Example
c = HamiltonianResidual(0.1:0.1:2.1)

c can then be used as an argument to Verbose(), which then goes as a keyword argument into evolve.
"""
struct HamiltonianResidual{V <: AbstractFloat} <: CurveInfoSnippet 
    timepoints::Vector{V}
end
CurveDistance(a::AbstractRange) = CurveDistance(a |> collect)
HamiltonianResidual(a::AbstractRange) = HamiltonianResidual(a |> collect)


"""
Construct as `v = Verbose(a::Vector{T}) where T <: CallbackCallable`

Example:
```v = Verbose([HamiltonianResidual(0.1:2.:10), CurveDistance(0.1:0.1:5)])
evolve(c::MDCProblem, ...; mdc_callback=[v, other_mdc_callbacks])
````
In this case, you would get REPL output as the curve evolved, indicating when the curve reached each distance in `0.1:0.1:5`, and indicating the residual on dHdu, which is ideally zero, at `0.1:0.1:5`.
"""
struct Verbose{T <: CurveInfoSnippet} <: CallbackCallable
    snippets::Vector{T}
end


"""
    ParameterBounds(ids, lbs, ubs)
    - ids are the indices of the model parameters that you want to bound
    - lbs are an array of lower bounds, with length == to indices
    - ubs are...well you can guess.
"""
struct ParameterBounds{I <: Integer,T <: Number} <: AdjustmentCallback
    ids::Vector{I}
    lbs::Vector{T}
    ubs::Vector{T}
end


Verbose() = Verbose([EmptyInfo()])
Verbose(snippet::EmptyInfo) = Verbose()
Verbose(snippet::T) where T <: CurveInfoSnippet = Verbose([snippet])

function (c::CurveDistance)(cp::CurveProblem, u, t, integ)
    @info "curve length is $t"
    nothing
end

function (h::HamiltonianResidual)(c::CurveProblem, u, t, integ)
    x = dHdu_residual(c, u, t, nothing)
    @info "dHdu residual = $x at curve length $t"
end

(e::EmptyInfo)(c, u, t, integ) = nothing



function (v::Verbose)(c::CurveProblem)
    to_call = map(v.snippets) do snippet
        (u, t, _integ) -> snippet(c, u, t, _integ)
    end
    return map(to_call, v.snippets) do each, snippet        
        FunctionCallingCallback(each; funcat=snippet.timepoints)
    end
end





"""
    DEPRECATED, use Verbose() instead.
    VerboseOutput(level=:low, times = 0:0.1:1.)
    Callback to give online info on how the solution is going, as the MDCurve evolves. activates at curve distances specified by times
"""
    function VerboseOutput(level=:low, times=0:0.1:1.)
    
        function affect!(integ)
            if level == :low 
                @info "curve length is $(integ.t)"
            end
            if level == :medium 
                @info "dHdu residual = "
            end
            if level == :high
        
            end
            return integ
        end
        return PresetTimeCallback(times, affect!) 
end



"""
    ParameterBounds(ids::Vector{Integer},lbs::Vector{Number},ubs::Vector{Number})
parameters[ids] must fall within lbs and ubs, where lbs and ubs are Arrays of the same size as ids.
Create hard bounds on the parameter space over which the minimally disruptive curve can trace. Curve evolution terminates if it hits a bound.
"""
    function (p::ParameterBounds)(c::CurveProblem)
        function condition(u, t, integrator)
            tests = u[p.ids]
            any(tests .< p.lbs) && return true
            any(tests .> p.ubs) && return true
            return false
        end
        return DiscreteCallback(condition, terminate!)
    end


