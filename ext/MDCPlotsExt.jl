module MDCPlotsExt

using MinimallyDisruptiveCurves
using Plots
using Plots.PlotMeasures # Required to unlock padding units like px, mm, or cm

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

    sys   = sample_sol.prob.p  
    chain = sys.cost.chain     
    θ₀    = sys.θ₀
    
    # 2. Reconstruct Continuous Time Domain Axis Safely
    min_t_bound = !isnothing(curve.negative_sol) ? -maximum(curve.negative_sol.t) : 0.0
    max_t_bound = !isnothing(curve.positive_sol) ? maximum(curve.positive_sol.t) : 0.0
    
    full_grid = collect(range(min_t_bound, stop=max_t_bound, length=density))
    sampled_states = [curve(t) for t in full_grid]
    
    # Isolate trailing arc-length coordinate scalar tracking variable from length representation
    N_params = (length(sampled_states[1])) ÷ 2
    
    # Safe dimension checking falling back smoothly on empty transformation chains
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
    if raw
        all_labels = [string(n) for n in transform_names(chain, sys.names)][active_indices]
    end
    labels_row = reshape(all_labels, 1, length(active_indices))

    # Calculate global tracking bounds across parameters for layout normalization
    y_min_bound, y_max_bound = minimum(Y_global), maximum(Y_global)
    margin_val = (y_max_bound - y_min_bound) * 0.05
    ylims_global = (y_min_bound - margin_val, y_max_bound + margin_val)

    # Hardcode explicit color vec to force perfect matches
    num_colors = length(active_indices)
    chosen_palette = reshape([Plots.palette(:auto)[i] for i in 1:num_colors], 1, num_colors)

    # Define custom asymmetric layout
    custom_layout = Plots.@layout [
        sim_pane
        bar_pane{0.33w} trajectory_pane
    ]

    # 3. Main Linear Animation Frame Sweep
    anim = Plots.@animate for (frame_idx, t_current) in enumerate(full_grid)
        
        # Initialize the figure with the custom layout and outer margin padding
        # FIX: Added top, bottom, left, and right margins (5mm) to stop edge cropping
        Plots.plot(
            layout = custom_layout, 
            size = (1100, 750), # Slightly increased height canvas to accommodate stack layers
            left_margin = 6mm, 
            right_margin = 6mm, 
            top_margin = 6mm, 
            bottom_margin = 6mm
        )

        state_current = curve(t_current)
        θ_transformed = state_current[1:N_params]
        θ_physical = MinimallyDisruptiveCurves.forward(chain, θ_transformed) 

        y_cursor = raw ? θ_physical[active_indices] : θ_transformed[active_indices]

        # --- PANEL 1: User Physics Simulation Sandbox (Entire Top Row) ---
        user_sim_func(1, θ_physical)
        Plots.plot!(subplot = 1, title = "Live System Behavior Profile")

        # --- PANEL 2: Instantaneous Value Deviation Bar Chart (Bottom Left, 1/3 Width) ---
        deltas = y_cursor .- θ₀_processed
        
        labels_as_series = reshape(all_labels, 1, length(all_labels))
        deltas_as_series = reshape(deltas, 1, length(deltas))

        Plots.bar!(
            labels_as_series, deltas_as_series,
            subplot = 2,                       
            orientation = :vertical,
            fillalpha = 0.7,
            linewidth = 1.2, linecolor = :match,
            color = chosen_palette,     
            bar_width = 0.95,            
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

        # --- PANEL 3: Structural Parameter Manifold Trace (Bottom Right, 2/3 Width) ---
        Plots.plot!(
            full_grid, Y_global,
            subplot = 3,
            linewidth = 1.0, linealpha = 0.15, label = false,
            color = chosen_palette     
        )

        Y_past = copy(Y_global)
        Y_past[(frame_idx + 1):end, :] .= NaN
        Plots.plot!(
            full_grid, Y_past,
            subplot = 3,
            linewidth = 3.0, linealpha = 0.9,
            label = labels_row, legend = :topleft,
            color = chosen_palette     
        )

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
