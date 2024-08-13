# ------------------------------------------------------------------------------
"""
    Node{d}()

Create a `d`-dim "empty" node whose levels are all 0 and indices are all -1. The
depth is set to 0. Such a node is used as a placeholder for invalid nodes.
"""
function Node{d}() where d
    return Node{d}(
        ntuple(i ->  0, d),
        ntuple(i -> -1, d),
    )
end # Node{d}


# ------------------------------------------------------------------------------
"""
    NodeValue{d}()

Create a `d`-dim "empty" node value whose nodal coefficient is `NaN`, hierarchi-
cal coefficient is `NaN`. Such a node value is used as a placeholder for invalid
node values.
"""
function NodeValue{d}() where d
    return NodeValue{d}(NaN, NaN)
end


# ------------------------------------------------------------------------------
"""
    AdaptiveSparseGrid{d}(max_depth::Int ; rtol::Float64 = 1e-2) where d

Create an empty `d`-dim adaptive sparse grid with the given `max_depth` and the
threshold `rtol` for adding a new node.

After construction, use `train!` to train the grid with a given function.
"""
function AdaptiveSparseGrid{d}(max_depth::Int ; rtol::Float64 = 1e-2) where d
    if d < 1; throw(ArgumentError("d must be >= 1")); end
    if max_depth < 2; throw(ArgumentError("max_depth must be >= 2")); end
    
    AdaptiveSparseGrid{d}(
        Dictionary{Node{d}, NodeValue{d}}(), # nv
        0,                                   # depth
        max_depth,                           # max_depth
        ntuple(i -> -1, d),                  # max_levels
        rtol,                                # rtol
        false                                # selfcontained
    )
end


# ------------------------------------------------------------------------------
"""
    RegularSparseGrid{d}(max_depth::Int, max_levels::NTuple{d,Int}) where d

Create a `d`-dim regular sparse grid with the given `max_depth` and `max_levels`
along each dimension, where `max_levels` is a tuple of integers indicating the
maximum levels along each dimension; `max_depth` is the depth of the tree, also
the so-called accuracy/refinement level of the grid.

## Notes
- All by-dimension level combinations `k` are kept such that sum(k) <= max_depth
+ d - 1.
"""
function RegularSparseGrid{d}(max_depth::Int, max_levels::NTuple{d,Int}) where d

    if any(max_levels .< 1); throw(ArgumentError("levels must be >= 1")); end

    # filter feasible levels & collect nodes
    nodes = Node{d}[]

    for ks in Iterators.product([1:l for l in max_levels]...)
        if sum(ks) <= (max_depth + d - 1)
            # given the levels by dimension, collect node indices of the levels
            itr = Iterators.product([
                get_all_1d_regular_index(k) for k in ks
            ]...)
            for i in itr
                push!(nodes, Node{d}(ks, i))
            end
        end
    end

    return RegularSparseGrid{d}(
        Dictionary{Node{d}, NodeValue{d}}(
            nodes,
            Vector{NodeValue{d}}(undef, length(nodes))
        ),
        max_levels,
        max_depth,
    )
end # RegularSparseGrid{d}


# ------------------------------------------------------------------------------
"""
    YellowPages{d}(
        G::AdaptiveSparseGrid{d} ;
        neighbortype::Symbol = :sparse
    ) where d

Create a `d`-dim yellow pages for the adaptive sparse grid `G` with the given
`neighbortype` (either `:sparse` or `:ghost`).

If `neighbortype == :sparse`, then the ASG must be self-contained.

## Notes
- I search sparse neighbors by checking the distance along each dimension. For
a large grid, this may be slow. May need to optimize this in the future.
- The ghost neighbors are directly defined based on the underlying mos-fined
grid such that it takes less time than the sparse neighbors.
"""
function YellowPages{d}(
    G::AdaptiveSparseGrid{d} ; 
    neighbortype::Symbol = :sparse
) where d
    
    n = length(G)
    A = Dictionary{Node{d},Int}(keys(G.nv), 1:n) # address

    # malloc
    L = Matrix{Node{d}}(undef, n, d)             # left
    R = Matrix{Node{d}}(undef, n, d)             # right

    # write the yellow pages
    if neighbortype == :sparse

        if !G.selfcontained
            throw(ArgumentError(
                "ASG must be self-contained for sparse neighbors"
            ))
        end

        for j in 1:d
            for (node, i) in pairs(A)

                # This inner loop can be parallelized for large number of nodes.
                # Given the current node, find all neighbors along the j-th dim
                # and compute the distance to the neighbors. Such neighbors must
                # have all the same (l,i) along other dimensions except the j-th
                # dimension. Meanwhile, remember to skip the node itself.

                allCandidates = Node{d}[]
                allDistances  = Float64[]
                bndflg        = get_boundaryflag(node, j)
                
                for node2 in keys(A)
                    if node2 == node; continue; end
                    if nodecmp_except(node, node2, j)
                        if !nodecmp_along(node, node2, j)

                            # distance = node2 - node, such that > 0 means
                            # right neighbor, < 0 means left neighbor
                            push!(allCandidates, node2)
                            push!(allDistances, get_dist_along(node, node2, j))

                        end
                    end
                end # node2

                # find the nearest left and right sparse neighbors
                if length(allCandidates) == 0
                    throw(ArgumentError(
                        "no sparse neighbors, check self-containedness"
                    ))
                elseif all(allDistances .<= 0) & (bndflg != 1)
                    throw(ArgumentError(
                        "no right sparse neighbors, check self-containedness"
                    ))
                elseif all(allDistances .>= 0) & (bndflg != -1)
                    throw(ArgumentError(
                        "no left sparse neighbors, check self-containedness"
                    ))
                end

                if bndflg == -1
                    # case: left boundary along the j-th dim
                    # do: only right neighbor possible; left neighbor is saved
                    #     as an invalid node

                    rloc   = argmin(allDistances[allDistances .> 0])
                    L[i,j] = Node{d}()
                    R[i,j] = allCandidates[allDistances .> 0][rloc]
                    
                elseif bndflg == 1
                    # case: right boundary along the j-th dim
                    # do: only left neighbor possible; right neighbor is saved
                    #     as an invalid node

                    lloc   = argmax(allDistances[allDistances .< 0])
                    L[i,j] = allCandidates[allDistances .< 0][lloc]
                    R[i,j] = Node{d}()

                else
                    # case: interior node
                    # do: both left and right neighbors are possible
                    #     find the nearest left and right neighbors
                    
                    lloc   = argmax(allDistances[allDistances .< 0])
                    rloc   = argmin(allDistances[allDistances .> 0])
                    L[i,j] = allCandidates[allDistances .< 0][lloc]
                    R[i,j] = allCandidates[allDistances .> 0][rloc]
                    
                end # if
            end # node
        end # j

    elseif neighbortype == :ghost
        maxlvls = get_maxlevels(G) # get maximum levels along each dimension
        for j in 1:d, (node, i) in pairs(A)
                L[i,j] = get_ghost_left(node, j, maxlvls[j])
                R[i,j] = get_ghost_right(node, j, maxlvls[j])
        end # j, (node, i)
    else
        throw(ArgumentError("invalid neighbor type: $neighbortype"))
    end
    return YellowPages{d}(A, L, R, neighbortype)
end # YellowPages{d}


# ------------------------------------------------------------------------------
"""
    LinearStencil{d}()

Create an empty `d`-dim linear stencil.
"""
function LinearStencil{d}() where d
    return LinearStencil{d}(Dictionary{Node{d},Float64}())
end # LinearStencil{d}


# ------------------------------------------------------------------------------
"""
    LinearStencil{d}(
        nodes::AbstractVector{Node{d}},
        weights::AbstractVector{Float64}
    ) where d

Create a `d`-dim linear stencil with the given `nodes` and `weights`.
"""
function LinearStencil{d}(
    nodes::AbstractVector{Node{d}},
    weights::AbstractVector{Float64}
) where d
    return LinearStencil{d}(Dictionary{Node{d},Float64}(nodes, weights))
end # LinearStencil{d}


# ------------------------------------------------------------------------------
"""
    Normalizer{d}()

Create a `d`-dim normalizer of hypercube `[0,1]^d`.
"""
function Normalizer{d}() where d
    return Normalizer{d}(ntuple(i -> 0.0, d), ntuple(i -> 1.0, d))
end # Normalizer{d}