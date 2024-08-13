# Opeartions on the Regular Sparse Grid structure
export get_ghost_stepsize



# ------------------------------------------------------------------------------
function Base.length(R::RegularSparseGrid{d})::Int where d
    return length(R.nv)
end # Base.length()
# ------------------------------------------------------------------------------
"""
    get_ghost_stepsize(R::RegularSparseGrid{d})::Vector{Float64} where d

Check `get_ghost_stepsize(G::AdaptiveSparseGrid{d})::Vector{Float64} where d`
for more details.
"""
function get_ghost_stepsize(R::RegularSparseGrid{d})::Vector{Float64} where d
    return Float64[1.0 / power2(l - 1) for l in R.max_levels]
end # get_ghost_stepsize()
# ------------------------------------------------------------------------------
"""
    convert2asg(
        R::RegularSparseGrid{d}; 
        rtol::Float64 = 1E-2
    )::AdaptiveSparseGrid{d} where d

Convert a regular sparse grid (RSG) to an adaptive sparse grid (ASG). The RSG
can be seen as a special case of the ASG. Such conversion does not require any
training, but simply copying the nodes and their values from the RSG to the ASG
such that more operations can be performed on the ASG.

## Notes
- In an RSG, (actual) depth = max depth = sum(max levels)
- Consider two ASG: `G1` and `G2`. Let `G1` be an ASG converted from an RSG of
depth `d`, max depth `dmax == d`, and max levels `maxlvls`. Let `G2` be an ASG
that was directly trained with the same `dmax`, while the tolerance is set to
make `d == dmax` and `maxlvls` are the same as `G1`. Then, `G1` and `G2` are
NOT the same due to the adaption. Usually, one could expect `G1` to have more
nodes than `G2`.
"""
function convert2asg(
    R::RegularSparseGrid{d} ;
    rtol::Float64 = 1E-2
)::AdaptiveSparseGrid{d} where d
    G = AdaptiveSparseGrid{d}(
        R.nv |> deepcopy,
        R.max_depth,
        R.max_depth,
        R.max_levels,
        rtol,
        false
    )
    G.selfcontained = is_selfcontained(G)
    return G
end # convert2asg()










