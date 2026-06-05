@recipe function f(curve::MinimallyDisruptiveCurves.MDCCurve; 
                   max_lines = nothing,   
                   mode = :absolute,      
                   raw = false,           
                   density = 200          
                  )
    
    sample_sol = !isnothing(curve.positive_sol) ? curve.positive_sol : curve.negative_sol
    if isnothing(sample_sol)
        error("Cannot plot an empty MDCCurve.")
    end
    
    sys   = sample_sol.prob.p  
    chain = sys.cost.chain     
    θ₀    = sys.θ₀
    
    t_grid = range(!isnothing(curve.negative_sol) ? minimum(curve.negative_sol.t) : 0.0, 
                   stop=!isnothing(curve.positive_sol) ? maximum(curve.positive_sol.t) : 0.0, 
                   length=density)
    sampled_states = [curve(t) for t in t_grid]
    
    N_params = length(sampled_states[1]) ÷ 2  # so this is transformed current parameters
    
    # Securely calculate output dimension by dry-running a single element
    dummy_forward = MinimallyDisruptiveCurves.forward(chain, sampled_states[1][1:N_params])
    out_dim = raw ? length(dummy_forward) : N_params
    
    Y = Matrix{Float64}(undef, length(t_grid), out_dim)
    
    # Non-allocating data unpacking via views
    for (t_idx, state) in enumerate(sampled_states)
        @views θ_current = state[1:N_params]
        if raw
            Y[t_idx, :] .= MinimallyDisruptiveCurves.forward(chain, θ_current)
        else
            Y[t_idx, :] .= θ_current
        end
    end
    
    θ₀_processed = raw ? MinimallyDisruptiveCurves.forward(chain, θ₀) : θ₀
    if mode == :relative
        for i in 1:out_dim
            Y[:, i] .-= θ₀_processed[i]
        end
    end
    
    # Filter for Top Movers
    active_indices = collect(1:out_dim)
    if !isnothing(max_lines) && max_lines < out_dim
        movements = [maximum(Y[:, i]) - minimum(Y[:, i]) for i in 1:out_dim]
        active_indices = sortperm(movements, rev=true)[1:max_lines]
        Y = Y[:, active_indices]
    end
    
    # Metadata Alignment
    title_suffix = mode == :relative ? " (Δ Change)" : (raw ? " (Raw Physical Units)" : " (Transformed Space)")
    title  --> "MDC Parameter Trajectories$title_suffix"
    xlabel --> "Arc Length Path Coordinate (t)"
    ylabel --> (mode == :relative ? "Δ Value" : "Value")
    
    # Safe fallback mapping for labels
    if raw
        labels = [string(sys.names[i]) for i in active_indices]
        # labels = [string("hi") for i in active_indices]
    else
        labels = [string(n) for n in transform_names(chain, sys.names)][active_indices]
    end
    
    label  --> reshape(labels, 1, length(active_indices))
    linewidth --> 2
    
    return collect(t_grid), Y
end
