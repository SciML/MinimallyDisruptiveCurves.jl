# DifferentiationInterface

For the stuff requiring finitediff or forwarddiff:

- instead of that, we add difftype as an argument
- the user then puts in difftype=AutoForwardDiff() or whatever.

Which stuff is this?

- `TransformCost`
- `TransformODESystem`


# ModelingToolkit dependencies

The functionality using MTK to transform is deprecated as this can be done in MTK itself now. Instead we should include scripts for this.

Specifically, there is a

- `change_independent_variable` that can reparamterise time
- `substitute_in_deriv_and_depvar` that can do a substitution of an MTK variable through ODE equations.

```julia
sub = Dict([k => m*ωn^2, c => 2ζ*m*ωn])
eqs_nd = substitute_in_deriv_and_depvar(eqs, sub)
```

EG `sub = Dict([x => log(abs(r))])` would replace the logabs transform


Transform_cost we can keep. But move to using DifferentiationInterface. EG

`transform_cost(C, AutoForwardDiff())`

so that the user separately loads the autodiff library of interest.


# Evolve and callbacks

We should just use the native callbackset type system and implement momentum readjustment etc as prespecificed diffeqcallbacks. This avoids the current mess of types and systems.


# Starting from scratch?

We fundamentally need:

MDCProblem
MDCSolution? Maybe we just want wrappers around the normal ODE sol struct.
Callbacks

And then niceties are transform_cost.


# Current logic

MDCProblem
