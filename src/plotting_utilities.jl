 @recipe function f(curve::MinimallyDisruptiveCurves.MDCCurve; 
                   max_lines = nothing,   # Int: limit to the N biggest movers
                   mode = :absolute,      # :absolute or :relative (delta from start)
                   raw = false,           # Bool: undo the system's TransformChain
                   density = 200          # Int: number of evaluation points for smooth curves
                  )
    
    # ----------------------------------------------------------------
    # 1. Access System Configurations via the Underlying Solutions
    # ----------------------------------------------------------------
    # We find the parent system via the ODE problem inside the solutions
    sample_sol = !isnothing(curve.positive_sol) ? curve.positive_sol : curve.negative_sol
    if isnothing(sample_sol)
        error("Cannot plot an empty MDCCurve.")
    end
    
    # Extract the system structures from the OrdinaryDiffEq problem workspace
    sys   = sample_sol.prob.p  # Assuming sys::MDCSystem was passed as p in ODEProblem
    chain = sys.cost.chain     # Access the TransformChain
    θ₀    = sys.θ₀
    
    # ----------------------------------------------------------------
    # 2. Reconstruct Continuous Time and States Axis
    # ----------------------------------------------------------------
    neg_t = !isnothing(curve.negative_sol) ? reverse(curve.negative_sol.t) : Float64[]
    pos_t = !isnothing(curve.positive_sol) ? curve.positive_sol.t : Float64[]
    
    if !isempty(neg_t)
        neg_t = -abs.(neg_t)
        pop!(neg_t) # Remove the duplicated 0.0 crossover point
    end
    full_t = vcat(neg_t, pos_t)
    
    # Create a smooth, dense sampling grid across the active span
    t_grid = range(minimum(full_t), stop=maximum(full_t), length=density)
    
    # Evaluate full states (parameters + costates) across the time grid
    sampled_states = [curve(t) for t in t_grid]
    N_total = length(sampled_states[1])
    N_params = N_total ÷ 2 # Isolate just the parameter dimensions
    
    # ----------------------------------------------------------------
    # 3. Parameter Transformation and Mode Calculations
    # ----------------------------------------------------------------
    # Pre-allocate parameter tracking matrix: coordinates (times, parameters)
    Y = Matrix{Float64}(undef, length(t_grid), N_params)
    
    for (t_idx, state) in enumerate(sampled_states)
        θ_current = state[1:N_params]
        
        # If raw == true, run the parameters backward through the TransformChain
        if raw
            # inverse_transform strips logs, scalings, etc., returning physical units
            Y[t_idx, :] .= inverse_transform(chain, θ_current)
        else
            Y[t_idx, :] .= θ_current
        end
    end
    
    # Establish the baseline coordinate for relative calculations
    θ₀_processed = raw ? inverse_transform(chain, θ₀) : θ₀
    
    if mode == :relative
        # Subtract the baseline configuration to show strictly change vectors (Δθ)
        for i in 1:N_params
            Y[:, i] .-= θ₀_processed[i]
        end
    end
    
    # ----------------------------------------------------------------
    # 4. Filter for Top "N" Biggest Movers
    # ----------------------------------------------------------------
    active_indices = collect(1:N_params)
    
    if !isnothing(max_lines) && max_lines < N_params
        # Quantify total movement per parameter as max value minus min value over the path
        movements = [maximum(Y[:, i]) - minimum(Y[:, i]) for i in 1:N_params]
        
        # Sort indices downward based on movement magnitude
        sorted_indices = sortperm(movements, rev=true)
        active_indices = sorted_indices[1:max_lines]
        
        # Slice matrix to isolate chosen tracking lines
        Y = Y[:, active_indices]
    end
    
    # ----------------------------------------------------------------
    # 5. Native Plot Canvas Attributes Configuration
    # ----------------------------------------------------------------
    title_suffix = mode == :relative ? " (Δ Change)" : (raw ? " (Raw Units)" : " (Transformed Space)")
    title  --> "MDC Parameter Trajectories$title_suffix"
    xlabel --> "Arc Length Path Coordinate (t)"
    ylabel --> (mode == :relative ? "Δ Value" : "Value")
    
    # Generate labels dynamically reflecting filtering matrices
    labels = ["θ_$i" for i in active_indices]
    label  --> reshape(labels, 1, length(active_indices))
    
    linewidth --> 2
    
    return t_grid |> collect, Y
end

using .RecipesBase

# This recipe activates specifically when you pass a Symbol as the second argument: 
# e.g., plot(mdc_curves, :costates)
@recipe function f(curve::MDCCurve, target::Symbol)
    
    # 1. Enforce that this recipe only runs for costates
    if target !== :costates
        error("Unknown plot target :$target. Did you mean :costates?")
    end

    # 2. Extract solution boundaries and set up the grid
    sample_sol = !isnothing(curve.positive_sol) ? curve.positive_sol : curve.negative_sol
    if isnothing(sample_sol)
        error("Cannot plot an empty MDCCurve.")
    end
    
    min_t = !isnothing(curve.negative_sol) ? minimum(curve.negative_sol.t) : 0.0
    max_t = !isnothing(curve.positive_sol) ? maximum(curve.positive_sol.t) : 0.0
    t_grid = range(min_t, stop=max_t, length=200)
    
    # 3. Evaluate and extract ONLY the costates (the second half of the state vector)
    sampled_states = [curve(t) for t in t_grid]
    N_total = length(sampled_states[1])
    N_params = N_total ÷ 2
    
    # Pre-allocate costate matrix (steps, N_params)
    L = Matrix{Float64}(undef, length(t_grid), N_params)
    for (t_idx, state) in enumerate(sampled_states)
        L[t_idx, :] .= state[(N_params + 1):end]
    end
    
    # 4. Define specific plot decorations for costates
    title   --> "MDC Costate Evolution"
    xlabel  --> "Arc Length Path Coordinate (t)"
    ylabel  --> "λ Value"
    
    # Create distinct labels for each multiplier component
    labels = ["λ_$i" for i in 1:N_params]
    label  --> reshape(labels, 1, N_params)
    
    linewidth --> 2
    
    # Return the clean continuous time axis and the isolated costate matrix
    return collect(t_grid), L
end
