module MDCPlotsExt

using MinimallyDisruptiveCurves
using Plots
using Plots.PlotMeasures 

import MinimallyDisruptiveCurves: MDCCurve, animate_mdc, transform_names

function MinimallyDisruptiveCurves.animate_mdc(
    curve::MDCCurve, 
    user_sim_func::Function; 
    fps = 15, 
    density = 100, 
    max_lines = 5, 
    raw = false
)
    # 1. Pipeline Verification
    sample_sol = !isnothing(curve.positive_sol) ? curve.positive_sol : curve.negative_sol
    if isnothing(sample_sol)
        error("Cannot animate an empty MDCCurve.")
    end

    mdc_sys = hasproperty(curve, :sys) ? curve.sys : sample_sol.prob.p
    
    # Unpack core mathematical structures
    chain = hasproperty(mdc_sys, :chain) ? mdc_sys.chain : mdc_sys.cost.chain
    θ₀    = mdc_sys.θ₀
    
    # 2. Reconstruct Continuous Time Domain Axis Safely
    min_t_bound = !isnothing(curve.negative_sol) ? minimum(curve.negative_sol.t) : 0.0
    max_t_bound = !isnothing(curve.positive_sol) ? maximum(curve.positive_sol.t) : 0.0
    
    full_grid = collect(range(min_t_bound, stop=max_t_bound, length=density))
    sampled_states = [curve(t) for t in full_grid]
    
    N_params = (length(sampled_states[1])) ÷ 2
    out_dim = (raw && !isempty(chain.ts)) ? chain.ts[end].full_dim : N_params
    
    Y_global = Matrix{Float64}(undef, length(full_grid), out_dim)
    for (t_idx, state) in enumerate(sampled_states)
        θ_current = state[1:N_params]
        Y_global[t_idx, :] .= raw ? MinimallyDisruptiveCurves.forward(chain, θ_current) : θ_current
    end

    θ₀_processed = raw ? MinimallyDisruptiveCurves.forward(chain, θ₀) : θ₀

    # Filter global parameter traces down to top N movers
    active_indices = collect(1:out_dim)
    if !isnothing(max_lines) && max_lines < out_dim
        movements = [maximum(Y_global[:, i]) - minimum(Y_global[:, i]) for i in 1:out_dim]
        active_indices = sortperm(movements, rev=true)[1:max_lines]
        Y_global = Y_global[:, active_indices]
        θ₀_processed = θ₀_processed[active_indices]
    end

    # Resolve active label naming mappings
    all_labels = String[]
    if hasproperty(mdc_sys, :names)
        base_names = mdc_sys.names 
        processed_names = raw ? transform_names(chain, base_names) : base_names
        all_labels = [string(n) for n in processed_names][active_indices]
    else
        all_labels = ["p_$i" for i in active_indices]
    end

    # Calculate global tracking bounds across parameters
    y_min_bound, y_max_bound = minimum(Y_global), maximum(Y_global)
    margin_val = (y_max_bound - y_min_bound) * 0.05
    ylims_global = (y_min_bound - margin_val, y_max_bound + margin_val)

    # Flatten palette elements into a 1D vector to prevent multi-subplot routing confusion
    num_colors = length(active_indices)
    color_array = [Plots.palette(:auto)[mod1(i, length(Plots.palette(:auto)))] for i in 1:num_colors]

    # Define custom asymmetric layout
    custom_layout = Plots.@layout [
        sim_pane
        bar_pane{0.33w} trajectory_pane
    ]

    # 3. Main Linear Animation Frame Sweep
    anim = Plots.@animate for (frame_idx, t_current) in enumerate(full_grid)
        
        Plots.plot(
            layout = custom_layout, 
            size = (1100, 750),
            left_margin = 6mm, right_margin = 6mm, 
            top_margin = 6mm, bottom_margin = 6mm
        )

        state_current = curve(t_current)
        θ_transformed = state_current[1:N_params]
        θ_physical = MinimallyDisruptiveCurves.forward(chain, θ_transformed) 

        y_cursor = raw ? θ_physical[active_indices] : θ_transformed[active_indices]

        # --- PANEL 1: User Physics Simulation Sandbox (Entire Top Row) ---
        user_sim_func(θ_physical)
        Plots.plot!(subplot = 1, title = "Live System Behavior Profile")

        # --- PANEL 2: Instantaneous Value Deviation Bar Chart ---
        deltas = vec(y_cursor .- θ₀_processed) # FIXED: Force 1D flat vector context

        Plots.bar!(
            all_labels, deltas,
            subplot = 2,                                
            orientation = :vertical,
            fillalpha = 0.7,
            linewidth = 1.2, linecolor = :match,
            color = color_array,     
            bar_width = 0.8,            
            label = false
        )
        
        max_delta = maximum(abs.(Y_global .- reshape(θ₀_processed, 1, length(θ₀_processed))))
        max_delta = max(1e-6, max_delta) 
        
        Plots.plot!(
            subplot = 2,                                
            title = "Instantaneous Parameter Shift (Δ)",
            ylabel = "Deviation from Nominal",
            ylims = (-max_delta * 1.2, max_delta * 1.2)
        )

        # --- PANEL 3: Structural Parameter Manifold Trace ---
        # FIXED: Plot matrix structures iteratively as flat 1D vectors to avoid column bleed across subplots
        for i in 1:size(Y_global, 2)
            Plots.plot!(
                full_grid, Y_global[:, i],
                subplot = 3,
                linewidth = 1.0, linealpha = 0.15, label = false,
                color = color_array[i]     
            )
        end

        # Isolate historical data trail segments
        Y_past = copy(Y_global)
        Y_past[(frame_idx + 1):end, :] .= NaN
        
        for i in 1:size(Y_past, 2)
            Plots.plot!(
                full_grid, Y_past[:, i],
                subplot = 3,
                linewidth = 3.0, linealpha = 0.9,
                label = all_labels[i],
                legend = :topleft,
                color = color_array[i]     
            )
        end

        Plots.scatter!(
            fill(t_current, length(active_indices)), y_cursor,
            subplot = 3,
            markersize = 6, markercolor = :red, label = false
        )

        Plots.plot!(
            subplot = 3,
            title = "Continuous Parameter Sweep (t = $(round(t_current, digits=2)))",
            xlabel = "Arc Length Coordinate (t)",
            ylabel = (raw ? "Physical Space Units" : "Transformed Optimization Units"),
            xlims = (min_t_bound, max_t_bound),
            ylims = ylims_global
        )
    end

    return anim
end

end # module
