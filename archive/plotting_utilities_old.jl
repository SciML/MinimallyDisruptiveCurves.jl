"""
plot recipe for ::MDCSolution
kwargs: pnames are array of parameter names
idxs: are parameter indices to plot
what ∈ (:trajectory, :final_changes) determines the plot type
"""
@recipe function f(mdc::MDCSolution; pnames = nothing, idxs = nothing, what = :trajectory)
    if idxs === nothing
        num = min(5, mdc.N)
        idxs = biggest_movers(mdc, num)
    end
    # if !(names === nothing)
    #     labels --> names[idxs]
    # end
    # ["hi" "lo" "lo" "hi" "lo"]

    tfirst = mdc.sol.t[1]
    tend = mdc.sol.t[end]

    layout := (1, 1)
    bottom_margin := :match

    if what == :trajectory
        @series begin
            if !(pnames === nothing)
                label --> reshape(pnames[idxs], 1, :)
            end
            title --> "change in parameters over minimally disruptive curve"
            xguide --> "distance"
            yguide --> "Δ parameters"
            distances(mdc), Δ(mdc)[idxs, :]'
        end
    end

    if what == :final_changes
        @series begin
            title --> "biggest changers"
            seriestype := :bar
            label --> "t=$tend"
            xticks --> (1:5, reshape(pnames[idxs], 1, :))
            xrotation --> 90
            Δ(mdc, tend)[idxs]
        end
        if tfirst < 0.0
            @series begin
                label --> "t=$tfirst"
                seriestype := :bar
                xticks --> (1:5, reshape(pnames[idxs], 1, :))
                xrotation --> 90
                Δ(mdc, tfirst)[idxs]
            end
        end
    end
end

"""
    output_on_curve(f, mdc, t)

Useful when building an animation of f(p) as the parameters p vary along the curve.
"""
function output_on_curve(f, mdc, t)
    return f(mdc(t)[:states])
end
