abstract type CurveInfoSnippet end
struct EmptyInfo <: CurveInfoSnippet end
struct CurveDistance <: CurveInfoSnippet end
struct HamiltonianResidual <: CurveInfoSnippet end


struct Verbose{T <: CurveInfoSnippet,S <: Real,V <: AbstractRange} 
    snippets::Vector{T}
    timepoints::Union{V{S},Vector{S}}
end

Verbose() = Verbose([EmptyInfo()], 0:0)
Verbose(snippet::EmptyInfo, times) = Verbose()
Verbose(snippet <: CurveInfoSnippet, times) = Verbose([snippet], times)

function (c::CurveDistance)(c::CurveProblem, u, t, integ)
    @info "curve length is $t"
    nothing
end

function (h::HamiltonianResidual)(c::CurveProblem, u, t, integ)
    x = dHdu_residual(c, u, t, nothing)
    @info "dHdu residual = $x"
end

(e::EmptyInfo)(c, u, t, integ) = nothing


# FunctionCallingCallback(func;
#                funcat=Vector{Float64}(),


function (v::Verbose)(c::CurveProblem)
    
    to_call = (u, t, _integ) -> map(_integ -> x(c, u, t, _integ), v.snippets)
    return FunctionCallingCallback(to_call; v.times)
end





"""
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
    function ParameterBounds(ids, lbs, ubs)
        function condition(u, t, integrator)
            tests = u[ids]
            any(tests .< lbs) && return true
            any(tests .> ubs) && return true
            return false
        end
        return DiscreteCallback(condition, terminate!)
    end


