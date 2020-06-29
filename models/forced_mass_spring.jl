

# @inline function if_else(condition::Bool, @nospecialize(trueval), @nospecialize(falseval))
#            return ifelse(condition, trueval, falseval)
# end

# ModelingToolkit.@register if_else(x, trueval, falseval)
# # This is needed to bring if_else into MTK's scope.
# @eval ModelingToolkit using ..Main: if_else

# nasty(i) = if_else(i > 1, 1, 0)
# @variables t
# nasty(t)



function MassSpringModel(input)
    
  @parameters t
  @parameters k,c,m
  @derivatives D'~t
  @variables pos(t) vel(t) inp(t)

  eqs = [D(pos) ~ vel,
        D(vel) ~ input(t) - (1/m)*(c*vel + k*pos),
  ]

  ps = [k,c,m] .=>  [1.,4.,1.] 
  ics = [pos, vel] .=> [1.,0.]
  od = ODESystem(eqs, t, first.(ics),first.(ps))
  # tspan = (0.,100.)
  # prob = ODEProblem(od, ics, tspan, ps)
  return od, ics, ps
end

# od,ic,ps = create(t->0)
# tspan = (0.,10.)
# prob = ODEProblem(od,ic,tspan,ps)
