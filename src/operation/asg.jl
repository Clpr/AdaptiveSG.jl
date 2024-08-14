export get_ghost_stepsize
export get_maxlevels
export append!
export is_selfcontained
export get_missing_boundarynodes, get_missing_boundarynodes_parallel
export make_selfcontained!


# ------------------------------------------------------------------------------
function Base.length(G::AdaptiveSparseGrid{d})::Int where d
    return length(G.nv)
end # Base.length()


# ------------------------------------------------------------------------------
"""
    get_ghost_stepsize(G::AdaptiveSparseGrid{d})::Vector{Float64} where d

Return the ghost mesh step size `h` of the underlying nodal grid, along each di-
mension of the adaptive sparse grid `G`. The ghost mesh step size is the mesh
step size of the largest level along each dimension. In a multi-dimensional case
the largest level is NOT equal to the depth of the ASG tree.

## Notes
- This function works only for a trained ASG.
"""
function get_ghost_stepsize(G::AdaptiveSparseGrid{d})::Vector{Float64} where d
    # get the max levels of each dimension; NOTES: max(level) != ASG tree depth
    return Float64[1.0 / power2(l - 1) for l in G.max_levels]
end # get_nodal_stepsize()


# ------------------------------------------------------------------------------
"""
    get_maxlevels(G::AdaptiveSparseGrid{d})::Vector{Int} where d

Return the maximum levels of the adaptive sparse grid `G` along each dimension.

## Notes
- If any dimension has no node, then the corresponding level is 0. This is impo-
ssible for a well-trained ASG. If this happens, check your ASG.
"""
function get_maxlevels(G::AdaptiveSparseGrid{d})::Vector{Int} where d
    maxlvls = fill(0, d)
    for node in keys(G.nv), j in 1:d
        maxlvls[j] = max(maxlvls[j], node.ls[j])
    end
    return maxlvls
end # get_maxlevels()


# ------------------------------------------------------------------------------
"""
    append!(
        G::AdaptiveSparseGrid{d}, 
        newnodes::Vector{Node{d}}
    ) where d

Append new nodes to the adaptive sparse grid `G.nv` without updating the corres-
ponding node values `(f,α)`, and all other meta info (fields of `G` except `nv`)

For a newly appended node `i`, `G.nv[i]` is initialized to a default invalid
`NodeValue{d}` object. The user should update the node values manually later.
"""
function Base.append!(
    G::AdaptiveSparseGrid{d}, 
    newnodes::Vector{Node{d}}
) where d
    for node in newnodes
        if !haskey(G.nv, node) & isvalid(node)
            insert!(G.nv, node, NodeValue{d}())
        end
    end
    return nothing
end # Base.append!()


# ------------------------------------------------------------------------------
"""
    is_selfcontained(G::AdaptiveSparseGrid{d})::Bool where d

Check if an ASG is self-contained, i.e. along each dimension, every interior and
boundary node has its left and right neighbor nodes.

See `get_missing_boundarynodes()` for more details.
"""
function is_selfcontained(G::AdaptiveSparseGrid{d})::Bool where d
    for j in 1:d, node in keys(G.nv)
        if !haskey(G.nv, perturb(node, j, 2, 0)); return false; end
        if !haskey(G.nv, perturb(node, j, 2, 2)); return false; end
    end
    return true
end # is_selfcontained()


# ------------------------------------------------------------------------------
"""
    get_missing_boundarynodes(G::AdaptiveSparseGrid{d})::Vector{Node{d}} where d

Return a vector of boundary nodes that are "missing" in the adaptive sparse grid
in the meaning of self-containedness. 

## Rules
- For each interior node `n1`, if along `j`-th dimension there is no left and/or
right neighbor node existing in the current `G`, then we think the corresponding
boundary node(s) are "missing". Here the "neighbor node" can be either interior
or boundary node.
- For each boundary node `n2` that is the left (or right) boundary along the `j`
-th dimension, if there is no right (or left) neighbor node existing in the cur-
rent `G`, then we think the corresponding right (or left) boundary node is "mis-
sing". Here the "neighbor node" can be either interior or boundary node.
- If every interior or boundary node has its left and right neighbor nodes along
every dimension, then the current `G` is thought to be "self-contained". This
function returns an empty vector then. (For a left boundary node, its left neig-
hbor does not existing but we technically use an invalid node to represent it,
such as in the yellow pages)

## Notes
- The first-time training of the ASG is not guaranteed to be self-contained bec-
ause some boundary nodes are not important for the interpolation given the user-
defined tolerance. This leads to as less nodes as possible. However, the ASG is
required to be self-contained if someone wants to use it to construct finite
difference operators/matrices e.g. in solving PDEs. In this case, one has to ap-
pend the missing boundary nodes to the ASG, `update!` the ASG, and then use it 
in PDE solvers.
- Notice that appending the missing boundary nodes to the ASG may significantly
increase the number of nodes in the ASG.
- This function is not parallelized since we expect the number of nodes in `G`
is not too large (millions at most). For medium and high dimension `G`, one can
use the parallelized version `get_missing_boundarynodes_parallel()` which uses
`@threads` to accelerate the traversal of nodes.

## Algorithm explaination
- We know a fact that: given all the other dimensions fixed, as long as there is
at least one node existing along the `j`-th dimension, the left and right bound-
ary nodes along the `j`-th dimension must exist for self-containedness. So we
do not need to really search for the actual neighbors (this is the job of the
`YellowPages{d}`). We just purtube every node along every dimension, and check
if such constructed boundary nodes exist in the current `G`. If not, the constr-
ucted boundary nodes are missing.
- We know a fact that: by fixing all the other dimensions, no matter how many
interior nodes exist along the `j`-th dimension, as long as the left and right
boundary nodes exist, then all nodes along the `j`-th dimension can properly de-
termine their left and right neighbors. This is the key property of the self co-
ntainedness.
- In this process, we do not need to distinguish the interior and boundary nodes
because the boundary nodes are also nodes in the `G`. The corresponding perturb-
ed boundary node is the node itself which will never be thought as missing.
"""
function get_missing_boundarynodes(
    G::AdaptiveSparseGrid{d}
)::Vector{Node{d}} where d
    newnodes = Node{d}[]

    # loop over each dimension 
    # Tips: `d` at the outer loop for easy parallelization extension
    # Tips: for a along-`j`-dim left boundary point `node`, `node == pl`. Simil-
    #       arly, for a right boundary point `node`, `node == pr`.
    for j in 1:d
        for node in keys(G.nv)
            pl = perturb(node, j, 2, 0)
            pr = perturb(node, j, 2, 2)
            if !haskey(G.nv, pl); push!(newnodes, pl); end
            if !haskey(G.nv, pr); push!(newnodes, pr); end
        end # node
    end # j
    return newnodes
end # get_missing_boundarynodes()


# ------------------------------------------------------------------------------
"""
    get_missing_boundarynodes_parallel(
        G::AdaptiveSparseGrid{d}
    )::Vector{Node{d}} where d

The parallelized version of `get_missing_boundarynodes()` which uses `@threads`
to accelerate the traversal of nodes. Check the non-parallel version for more
details.

This parallelized version is useful when the dimension * number of nodes is very
large. For example, for a 10-dim ASG with 10^4 nodes, the non-parallel version
is enough. But for a 100-dim ASG with 10^5 nodes, the parallel version is more
efficient.
"""
function get_missing_boundarynodes_parallel(
    G::AdaptiveSparseGrid{d}
)::Vector{Node{d}} where d
    nthread           = Threads.nthreads()
    newnodes_bythread = Dict{Int, Vector{Node{d}}}()
    for i in 1:nthread
        newnodes_bythread[i] = Node{d}[]
    end

    for j in 1:d
        Threads.@threads for node in keys(G.nv)
            tid = Threads.threadid()
            pl = perturb(node, j, 2, 0)
            pr = perturb(node, j, 2, 2)
            if !haskey(G.nv, pl); push!(newnodes_bythread[tid], pl); end
            if !haskey(G.nv, pr); push!(newnodes_bythread[tid], pr); end
        end # node
    end # j

    return vcat(newnodes_bythread...)
end # get_missing_boundarynodes_parallel()


# ------------------------------------------------------------------------------
"""
    make_selfcontained!(G::AdaptiveSparseGrid{d})::Nothing where d

Append the missing boundary nodes to the adaptive sparse grid `G` and update the
`selfcontained` field of `G` to `true`. This function is simply a wrapper of the
workflow: 

`get_missing_boundarynodes()` -> `append!()`.
"""
function make_selfcontained!(
    G::AdaptiveSparseGrid{d}
)::Nothing where d
    newnodes = get_missing_boundarynodes(G)
    append!(G, newnodes)
    return nothing
end # make_selfcontained!()