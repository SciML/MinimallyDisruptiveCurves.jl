
soft_heaviside(t, nastiness, step_time) = 1 / (1 + exp(nastiness * (step_time - t)))
soft_heaviside(nastiness, step_time) = t -> soft_heaviside(t, nastiness, step_time)


