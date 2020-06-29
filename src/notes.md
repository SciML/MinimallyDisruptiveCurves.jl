### To dos for v1.0

-[x] log problem functionality. recover parameter names if they exist and store them.
- [x] callback for adjusting costates when the res gets too big
- [x] parameter constraints functionality: provide box constraints?
- [x] free parameters structure: free only the labelled parameters
- [(x)] build a gradient function from finite differences if there is no alternative (already in build_loss_objective though?)
- [x] or just do multithreading!
- [x] Injection loss
- [ ] second order sensitivities for initial directions



### Plotting and evolve wrapper

Idea is to wrap sol.t in a DataFrame! (the ! means solutions are not copied)
Is there a way to do this withouit explicitly using DataFrames?
Yes: we can use a NamedTuple. Then the user can do 
> df = DataFrame(NamedTuple)

But a plot recipe might be easier: then I can make a custom visualisation.



evolve wrapper should multithread if it is a two sided curve. 




### Loss functions from the ground up DEPRECATED

I intend to replace and extend the DiffEqParamEstim code for buliding a loss objective. This means that the extra dependency is not required. It also gets rid of apparent instability issues with Zygote, as I can switch between ForwardDiff, Zygote, and AdjointSensitivity as appropriate.

I want the user to have the option of FD, adjoint (and maybe Zygote).

I want ensemble functionality built in, so we can sum and parallelise over loss functions.


One issue with Zygote=type gradients: you get the gradient of the whole solution

### Callbacks (Done)
state and momentum readjustment
FunctionCallingCallback (verbose output at t=0.1)
Parameter Constraints
costs over solution

cb = CallbackSet(cb1,cb2,cb3) 


rescond needs momentum, cost, theta0, tolerance, N
costate_affect needs momentum, cost, theta0
state_affect needs momentum cost(and grad) and theta0

print output callback: PeriodicCallback or something like that


- Parallelise running in both directions
- re-export DiffEqParamEstim.jl, or get rid of it as a dependency altogether?
- Figure out how to sync my WeightDrifter.jl toolbox loss functions with this easily
- Parallelise curve solving as well as running in both directions...maybe the MonteCarlo stuff in DiffEq.jl has good code 

- momentum_readjust
- state_readjust

- LIBRARY OF INPUT FEATURES: SUMMING LOSS OBJECTIVES


### Extras

Think about incorporating a terminal point-there may be interest. This could be by solving a BVP, or otherwise by adding something to the line integrand: eg the constraint that the curve gets closer to the terminal point. Could call this function 'bridge()'tur


##### Base functionality
Take a cost function in the style of build_loss_objective in DiffEqParamEstim.jl
- so cost(p) gives cost
- cost(p,g) gives cost, and mutates g to give gradient


Then evolve a minimally disruptive curve with this choice of cost function.


##### Extra functionality
Note that build_loss_objective takes a function loss(sol), and turns it into an object like cost() described above
- I want a way of taking multiple build_loss_objectives, and summing their output to create a new objective. This is necessary whenh we have a library of inputs.

- I also want a way of making an array of ODEProblems, given a library of inputs u(t). I can then sum the build_loss_objective() on each problem.
NOTE that build_loss_objective takes DiffEqBase.DEProblem. A subtype of this is EnsembleProblem. So I can probably do this stuff for ensemble problems using the existing functionality

### How DiffEqParamEstim works

DiffEqParamEstim has a 'build_loss_objective' funciton. You put in either your own function loss(sol::ODEsoln) or you use the provided ones:L2Loss, LogLikeLoss, etc

Either way, build_loss_objective returns a struct::DiffEqObjective.

struct DiffEqObjective{F,F2} <: Function
  cost_function::F
  cost_function2::F2
end

This then goes into the Optim.jl optimize function:
optimize(d::T, initial_x::AbstractArray, options::Options) where T<:AbstractObjective
T is a DiffEqObjective



### How to structure my code

I don't actually have to use anything from Optim.jl, but I can get motivation from it, as structurally it does the same thing as my package wants to do.


BELOW IS NOT CORRECT: DiffEqParamEstim ignores it
I just need my package to work with a loss function struct that obeys the same laws as d::T in optimize. IE it has the same API for taking gradients, etc. Then any d::T that goes into optimize, can also go into my evolve. The API is provided at the NLSolversBase.jl github page:
https://github.com/JuliaNLSolvers/NLSolversBase.jl

All I need to do is work with the loss objectives, not build them myself


Optimize has a load of methods and fallbacks for all the use cases on what gradient the user supplies/doesn't supply, so it can exploit as much as possible. This is in
https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/optimize/interface.jl

Meanwhile, once all the user options are sorted out, the main loop is in
https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/optimize/optimize.jl


Methods required for d::T in optimize:

tr = OptimizationTrace{typeof(value(d)), typeof(method)}()
trace!(tr, d, state, iteration, method, options, _time-t0)
update_state!(d, state, method)
update_g!(d, state, method) # TODO: Should this be `update_fg!`?


It needs an init direction. so we need a get_init_dirs(loss_function) function

Another idea:

function logged_problem(p::ODEProblem)
make new problem that does exp(p) then vector firled of p
return new problem
end


cost_function2 = function (p,grad)  (in DiffEqParamEstim)
changes the grad in place. and returns the cost.



From Optim.jl:
MDCurve <: FirstOrderOptimizer 
MDCurveState <: AbstractOptimizerState
update_state!(d, m::MDCurve, ms::MDCurveState)
trace pops up a lot: it holds information on the last update for momentum-y methods


#### Injection loss notes


"""

Final verdict: 
- Injection cost for all states / all states where parameters are in the derivatives equates to Two stage method. Can put this in the docs.
- Injection cost for a couple of states, but not all states with parameters...why would we even want this? Would add complication to docs and functionality for no purpose. So let's kill it.


Injection loss is similar to Two-stage method. How much derivative to I have to inject to make the parameterised trajectory follow the nominal trajectory.

Two-stage method

\dot{x} = f(x,θ)
\dot{x}* = f*(x)

- Integral along x^*(t) of (f(x,θ) - f^*(x))^2
- Note that we have to add an injection of f(x,θ) - f^*(x) to every state. But we only - add to the cost those states indexed by ids

In the case of a conductance based model where only the maximal conductances are parameterised: the injection to the non-voltage states is a priori zero. Thus we can think about a 'current' injection.

Now what if I want the current injection loss (ie a particular state) in a situation where ion-channel parameters (ie attached to derivatives of other states) are nonzero. Then I actually have to integrate along x(θ,t), and thus solve the system. So the special case here is where I consider injection to a **subset** of the states, but still want those states to follow the nominal trajectory.



So what do I want? Full
1) Injection cost (generic, requires solve)
2) Injection cost where variable parameters are attached only to the injected states 
    -> TwoStageMethod (vanilla: if conscientious can enforce zero divergence on non-injected states, which we know apriori to be true)
3) Generic two stage cost (i.e. injection cost on all states)


An issue: two stage method in DiffEqParamEstim uses SMOOTHED estimates of the trajectories. This is great for data. Not great for nominal problem.


dx  = prob.f(x,p,t) + nomf(x)[ids] - prob.f(x,p,t)[ids]
dx[ids]    = nomf(x)[ids]
dx[!ids] = f(x,p,t)
x[ids](t) = x^*(t)

injectionLoss -> prob
prob -> sol
L2Loss(sol)

so buid_loss_objective with that L2Loss

build_loss_objective:
prob -> sol
cost = sol -> (lossf, sol) -> loss
costg = xxx with gradient

"""
