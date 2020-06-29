"""
Utilities to find and plot the biggest changing parameters
"""


"""
    find parameter indices of the biggest changing parametesr in the curve
"""
function biggest_movers(mdc::MinimallyDisruptiveCurve, num::Integer; rev=false)
    diff = trajectory(mdc)[:,end] - trajectory(mdc)[:,1]
    ids = sortperm(diff, by=abs, rev=!rev)
    ids = ids[1:num]
end

