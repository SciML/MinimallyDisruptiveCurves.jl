
function vf_forced(δx,x,p,t, inp)
    # vector field for mass spring system. appended (last state) is C(t), where cost functional J = ∫ C(t) dt. I used a linear spring and a cubic spring, in order that there exists a bistable solution with unstable f.p. at 0 and stable at ± something, for some parameters.
  pos,vel = x
  k,c,m = p
  δx[1] = vel
  δx[2] = inp(t) - (1/m)*(c*x[2] + k*x[1])
  return δx
end
    

function vf(δx,x,p,t)
    # vector field for mass spring system. appended (last state) is C(t), where cost functional J = ∫ C(t) dt. I used a linear spring and a cubic spring, in order that there exists a bistable solution with unstable f.p. at 0 and stable at ± something, for some parameters.
  pos,vel = x
  k,c,m = p
  δx[1] = vel
  δx[2] = 0 - (1/m)*(c*x[2] + k*x[1])
  return δx
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p0 = [1.,4.,1.]
prob = ODEProblem(vf, u0, tspan, p0)
