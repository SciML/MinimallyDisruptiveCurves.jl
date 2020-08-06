

"""
makes a soft analogue of the heaviside step function. useful for inputs to differential equations, as it's easier on the numerics.
"""
soft_heaviside(t, nastiness, step_time) = 1 / (1 + exp(nastiness * (step_time - t)))
soft_heaviside(nastiness, step_time) = t -> soft_heaviside(t, nastiness, step_time)


get_ids_names(opArray) = repr.(opArray)
