"""
Utilities to find and plot the biggest changing parameters
"""


"""
    find parameter indices of the biggest changing parametesr in the curve
"""
function biggest_movers(curve_sol, num::Integer; rev=false)
    N =Int(0.5*length(curve_sol.u[end]))
    diff = curve_sol.u[end][1:N] - curve_sol.u[1][1:N]
    ids = sortperm(diff, by=abs, rev=!rev)
    ids = ids[1:num]
end


"""
    add solutions in positive and negative directions
"""
function merge_sols()
    #easier to make a new structure type that stores/refs the sols, and calls them. rather than 
end