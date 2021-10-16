## How MDC works right now:

```julia
specify_curve()
specify_curve(log_cost, newp0, newdp0, mom, span)

cb1 = ParameterBounds([1,3], [-10.,-10.], [10.,10.])
cb2 = VerboseOutput(:low, 0.1:2.:10)
cb = CallbackSet(cb1, cb2);
@time mdc = evolve(eprob, Tsit5; callback=cb)
```

Basically curveProblem is not overloaded. It represents and MDC problem.

What I want to do is add an extra layer of abstraction so that MDC and MDC_Jumped and other types of problem can come in there.

Current:

AbstractCurveSpec is for the curveProblem types
AbstractCurve is for the solution types

Proposed:
CurveProblem is the old AbstractCurveSpec
MDCProblem <: CurveProblem
MDCProblemJumped <: CurveProblem




Current:
evolve(c::Curveproblem):
    1. make ODEProblem from c
    2. add provided callbacks. add momentum callback if not NaN (I think momentum should go in the curveproblem)
    3. Give default solmethod Tsit5 if not specified
    4. Spans: if two sided, then make two curve problems, run on threads and merge
    5. Turn solution into Curve type

Proposed:
evolve(c::MDCProblem) or evolve(c::JumpedMDCProblem)



    1. replace make_ODEProblem with a functor
    2. outsource dealing with the callbacks to a new function
    3. Put dealing with the spans in the functor for making the ODEProblem. Give a tuple of problems, of length 1 or 2. Use tmap for solving.
    4. 

add remake function to change problem statistics



### What do I want in solve vs curveProblem:

solve should have algorithm eg Tsit5


## Readjustment current:

readjustment:
    condition = :res
        cond = rescond
    condition = :cost
        cond = costcond
    readjust = :state
        affect = state_affect
    readjust = :momentum
        affect = costate_affect
cb = DiscreteCallback(cond, affect)

## Issues

something is going very weird with using old arrays:

with play.jl (the lotka volterra) each time i run evolve i get a progressively weidrer curve. only if the spans are two sided.