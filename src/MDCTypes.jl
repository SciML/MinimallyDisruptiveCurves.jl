abstract type CurveProblem end
abstract type CurveModifier end
abstract type WhatJump <: CurveModifier end
abstract type WhatDynamics <: CurveModifier end

struct JumpStart{F <: AbstractFloat} <: WhatJump
    jumpsize::F
end
struct ZeroStart <: WhatJump end

function make_spans(c::CurveProblem, span)
    return make_spans(c::CurveProblem, span, isjumped(c))
end


struct MDCDynamics <: WhatDynamics end

dynamics(c::CurveProblem) = dynamics(c::CurveProblem, whatdynamics(c))
build_cond(c::CurveProblem, r, tol) = build_cond(c, r, tol, whatdynamics(c))
dHdu_residual(c::CurveProblem, u, t, p) = dHdu_residual(c, u, t, p, whatdynamics(c))
build_affect(c::CurveProblem, affect) = build_affect(c, affect, whatdynamics(c))




"""
For callbacks to tune MD Curve
"""
abstract type ConditionType end
struct ResidualCondition <: ConditionType end
struct CostCondition <: ConditionType end 

abstract type CallbackCallable end
abstract type AdjustmentCallback <: CallbackCallable end

"""
    MomentumReadjustment(tol::AbstractFloat, verbose::Bool)
Ideally, dHdu = 0 throughout curve evolution, where H is the Hamiltonian/momentum, and u is the curve velocity in parameter space. Numerical error integrates and prevents this. This struct readjusts momentum when `abs(dHdu) > tol`, so that `dHdu = 0` is recovered. 
"""
struct MomentumReadjustment{T <: AbstractFloat} <: AdjustmentCallback
    tol::T
    verbose::Bool
end

"""
    Terminates curve evolution when the cost exceeds the momentum
"""
struct TerminalCond <: AdjustmentCallback end
MomentumReadjustment(a; verbose=false) = MomentumReadjustment(a, verbose)




"""
Experimental. See documentation for MomentumReadjustment. This acts the  same, but instead of modifying the momentum, it modifies the state of the curve (i.e. current parameters) itself, by doing gradient descent to minimise the cost function, subject to the constraint that the distance from the initial parameters does not decrease. 
"""
struct StateReadjustment{T <: AbstractFloat} <: AdjustmentCallback
    tol::T
verbose::Bool
end
StateReadjustment(a; verbose=false) = StateReadjustment(a, verbose)


abstract type AffectType end
struct StateAffect <: AffectType end
struct CostateAffect <: AffectType end