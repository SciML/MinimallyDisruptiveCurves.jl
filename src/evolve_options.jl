
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


