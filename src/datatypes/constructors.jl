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
function AdaptiveSparseGrid{d}(
    max_depth::Int ; 
    rtol::Float64  = NaN,
    atol::Float64  = NaN,
    use_rtol::Bool = false,
) where d
    if d < 1; throw(ArgumentError("d must be >= 1")); end
    if max_depth < 2; throw(ArgumentError("max_depth must be >= 2")); end
    if use_rtol && isnan(rtol); throw(ArgumentError("rtol needed")); end
    if (!use_rtol) && isnan(atol); throw(ArgumentError("atol needed")); end

    AdaptiveSparseGrid{d}(
        Dictionary{Node{d}, NodeValue{d}}(), # nv
        0,                                   # depth
        max_depth,                           # max_depth
        ntuple(i -> -1, d),                  # max_levels
        use_rtol ? rtol : NaN,               # rtol
        use_rtol ? NaN  : atol,              # atol
        use_rtol,                            # use_rtol
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
    Normalizer{d}()

Create a `d`-dim normalizer of hypercube `[0,1]^d`.
"""
function Normalizer{d}() where d
    return Normalizer{d}(ntuple(i -> 0.0, d), ntuple(i -> 1.0, d))
end # Normalizer{d}