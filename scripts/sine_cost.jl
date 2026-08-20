using MinimallyDisruptiveCurves,Plots,OrdinaryDiffEq

include("./cost_functions.jl")


A = 1.0
ω = 1.0

cost = make_sin_cost_function(A, ω)

initial_direction = [1.0,1.0]

H = 1.0

sys = MDCProblem(cost, [0.0, 0.0], initial_direction, H; names = [:x, :y])

stabilizer = mdc_momentum_readjustment(sys; tol = 1.0e-3)
logger = mdc_verbose_callbacks(sys, range(-2.0, 5.0, 20))

my_pipeline = CallbackSet(stabilizer, logger...)
log_pipeline = CallbackSet(logger...)

mdc_curve = MDCSolve(sys, span = MDCSpan(0.0, 10.0), callback = nothing, use_pi_control = true, pi_params = [0.3, 0.01])

xs =  getindex.(mdc_curve.positive_sol.u, 1)
ys =  getindex.(mdc_curve.positive_sol.u, 2)

println("total cost: $(mdc_curve.positive_sol.u[end][end])")
scatter(xs, ys,
    label="label",
    title="curve",
    xlabel="x",
    ylabel="y"
)