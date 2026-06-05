
# ====================================================================
# TARGET BRANCH A: Standard Continuous Parameter Sweep Line Trace
# ====================================================================
@recipe function f(curve::MinimallyDisruptiveCurves.MDCCurve; 
                   max_lines = nothing,   # Int: limit to the N biggest movers
                   mode = :absolute,      # :absolute or :relative (delta from start)
                   raw = false,           # Bool: true unpacks optimization params back to Physical space
                   density = 200          # Int: number of evaluation points for smooth curves
                  )
    
    # 1. Access System Configurations via the Underlying Solutions
    sample_sol = !isnothing(curve.positive_sol) ? curve.positive_sol : curve.negative_sol
    if isnothing(sample_sol)
        error("Cannot plot an empty MDCCurve.")
    end
    
    sys   = sample_sol.prob.p  
    chain = sys.cost.chain     
    θ₀    = sys.θ₀
    
    # 2. Reconstruct Continuous Time and States Axis Safely
    min_t = !isnothing(curve.negative_sol) ? minimum(curve.negative_sol.t) : 0.0
    max_t = !isnothing(curve.positive_sol) ? maximum(curve.positive_sol.t) : 0.0
    
    t_grid = range(min_t, stop=max_t, length=density)
    sampled_states = [curve(t) for t in t_grid]
    
    N_total = length(sampled_states[1])
    N_params = N_total ÷ 2 
    
    # 3. Parameter Transformation and Mode Calculations
    # If raw=true, we project to Full Physical Dimension space via forward()
    out_dim = raw ? chain.ts[end].full_dim : N_params
    Y = Matrix{Float64}(undef, length(t_grid), out_dim)
    
    for (t_idx, state) in enumerate(sampled_states)
        θ_current = state[1:N_params]
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
    
    # 4. Filter for Top "N" Biggest Movers
    active_indices = collect(1:out_dim)
    if !isnothing(max_lines) && max_lines < out_dim
        movements = [maximum(Y[:, i]) - minimum(Y[:, i]) for i in 1:out_dim]
        sorted_indices = sortperm(movements, rev=true)
        active_indices = sorted_indices[1:max_lines]
        Y = Y[:, active_indices]
    end
    
    # 5. Native Plot Canvas Attributes Configuration
    title_suffix = mode == :relative ? " (Δ Change)" : (raw ? " (Raw Units)" : " (Transformed Space)")
    title  --> "MDC Parameter Trajectories$title_suffix"
    xlabel --> "Arc Length Path Coordinate (t)"
    ylabel --> (mode == :relative ? "Δ Value" : "Value")
    
    # Dynamic Metadata Extraction: Ensure labels align with output dimensional context
    if raw
        labels = [string(n) for n in transform_names(chain, sys.names)][active_indices]
    else
        labels = [string(sys.names[i]) for i in active_indices]
    end
    
    label     --> reshape(labels, 1, length(active_indices))
    linewidth --> 2
    
    return t_grid |> collect, Y
end

# ====================================================================
# TARGET BRANCH B: Categorical Multi-Target Diagnostics Subplots
# ====================================================================
@recipe function f(curve::MinimallyDisruptiveCurves.MDCCurve, target::Symbol; 
                   max_lines = 5, 
                   raw = false, 
                   density = 200)
    
    sample_sol = !isnothing(curve.positive_sol) ? curve.positive_sol : curve.negative_sol
    if isnothing(sample_sol)
        error("Cannot plot an empty MDCCurve.")
    end
    
    sys   = sample_sol.prob.p  
    chain = sys.cost.chain     
    
    min_t = !isnothing(curve.negative_sol) ? minimum(curve.negative_sol.t) : 0.0
    max_t = !isnothing(curve.positive_sol) ? maximum(curve.positive_sol.t) : 0.0
    t_grid = range(min_t, stop=max_t, length=density)
    
    sampled_states = [curve(t) for t in t_grid]
    N_total = length(sampled_states[1])
    N_params = N_total ÷ 2
    
    # --- TARGET 1: Standalone Cost Profile ---
    if target === :cost
        costs = [sys.cost(state[1:N_params]) for state in sampled_states]
        
        title     --> "MDC Objective Cost Verification"
        xlabel    --> "Arc Length Path Coordinate (t)"
        ylabel    --> "Objective Cost V(θ)"
        label     --> "Path Cost"
        linewidth --> 2
        
        return collect(t_grid), costs

    # --- TARGET 2: Standalone Costate Evolution ---
    elseif target === :costates
        L = Matrix{Float64}(undef, length(t_grid), N_params)
        for (t_idx, state) in enumerate(sampled_states)
            L[t_idx, :] .= state[(N_params + 1):end]
        end
        
        title     --> "MDC Costate Evolution"
        xlabel    --> "Arc Length Path Coordinate (t)"
        ylabel    --> "λ Value"
        
        labels    = ["λ_($n)" for n in sys.names]
        label     --> reshape(labels, 1, N_params)
        linewidth --> 2
        
        return collect(t_grid), L

    # --- TARGET 3: Stacked Diagnostic Dashboard Panels ---
    elseif target === :summary
        out_dim = raw ? chain.ts[end].full_dim : N_params
        Y = Matrix{Float64}(undef, length(t_grid), out_dim)
        
        for (t_idx, state) in enumerate(sampled_states)
            θ_current = state[1:N_params]
            Y[t_idx, :] .= raw ? MinimallyDisruptiveCurves.forward(chain, θ_current) : θ_current
        end
        
        movements = [maximum(Y[:, i]) - minimum(Y[:, i]) for i in 1:out_dim]
        active_indices = sortperm(movements, rev=true)[1:min(max_lines, out_dim)]
        Y_filtered = Y[:, active_indices]
        costs = [sys.cost(state[1:N_params]) for state in sampled_states]
        
        layout --> (2, 1)
        link   --> :x  
        
        # --- Subplot panel 1: Structural Parameters ---
        @series begin
            subplot   --> 1
            title     --> "MDC Trajectory Analysis Summary"
            ylabel    --> (raw ? "Raw Units" : "Transformed Value")
            
            if raw
                labels = [string(n) for n in transform_names(chain, sys.names)][active_indices]
            else
                labels = [string(sys.names[idx]) for idx in active_indices]
            end
            
            label     --> reshape(labels, 1, length(active_indices))
            linewidth --> 2
            
            collect(t_grid), Y_filtered
        end
        
        # --- Subplot panel 2: Background Evaluation Costs ---
        @series begin
            subplot   --> 2
            ylabel    --> "Objective Cost V(θ)"
            xlabel    --> "Arc Length Path Coordinate (t)"
            label     --> "Landscape Cost Profile"
            linewidth --> 2
            linecolor --> :black
            fillrange --> 0.0          
            fillalpha --> 0.1
            
            collect(t_grid), costs
        end

        return nothing
    else
        error("Unknown plot target symbol: :$target. Supported choices are :cost, :costates, or :summary.")
    end
end

# ====================================================================
# UTILITY HELPER: Standalone Cost Verification Slices
# ====================================================================
function cost_profile(curve::MinimallyDisruptiveCurves.MDCCurve; density=100)
    sample_sol = !isnothing(curve.positive_sol) ? curve.positive_sol : curve.negative_sol
    if isnothing(sample_sol)
        error("Cannot process an empty MDCCurve.")
    end
    sys = sample_sol.prob.p
    
    min_t = !isnothing(curve.negative_sol) ? minimum(curve.negative_sol.t) : 0.0
    max_t = !isnothing(curve.positive_sol) ? maximum(curve.positive_sol.t) : 0.0
    
    t_grid = range(min_t, stop=max_t, length=density)
    N_params = length(sys.θ₀)
    costs = Vector{Float64}(undef, length(t_grid))
    
    for (i, t) in enumerate(t_grid)
        θ_current = curve(t)[1:N_params]
        costs[i] = sys.cost(θ_current) 
    end
    
    return collect(t_grid), costs
end
