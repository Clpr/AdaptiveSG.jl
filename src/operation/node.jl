export isvalid        # node validation
export nodecmp, nodecmp_except, nodecmp_along # to find potential neighbors
export get_x          # level-index pair representation to node value
export fraction2li    # fraction representation to level-index pair
export perturb        # to modify a node along a dimension
export sortperm, sort, sort!                       # sort nodes
export get_child_left, get_child_right, get_parent # for tree growth
export get_ghost_left, get_ghost_right # for ghost nodes
export get_boundaryflag                # for boundary check
export get_all_1d_nodal_nodes_in_order # for quick creating nodal grid for test
export get_dist_along # to compute mesh step size along a dimension

export get_all_1d_regular_index # for regular sparse grid index
export glue # to raise the dimension of the nodes

# ------------------------------------------------------------------------------
"""
    isvalid(node::Node{d})::Bool where d

Check if the given `d`-dim node is valid.

This function is used to exclude the out-of-boundary nodes grown by (l=2,i=0) to
wards the left and (l=2,i=2) towards the right.

## Rules
- for all dimensions, level > 0
    - for a dimension, if level == 1, then index == 1 only
    - for a dimension, if level == 2, then index == 0 or 2 only
    - for a dimension, if level > 2, then index is odd and positive
- `node.depth > 0` must be true
"""
function isvalid(node::Node{d})::Bool where d
    # check dimension
    if d < 1; return false; end

    # check: levels & indices by dimension
    for j in 1:d
        @inbounds if node.ls[j] < 1
            return false
        elseif (node.ls[j] == 1)
            if node.is[j] != 1
                return false
            end
        elseif node.ls[j] == 2
            if (node.is[j] != 0) & (node.is[j] != 2)
                return false
            end
        else
            if (node.is[j] < 1) | (!isodd(node.is[j]))
                return false
            end
        end # if
    end # j

    # check: depth
    if node.depth < 1; return false; end
    if node.depth != nodedepth(node.ls); return false; end

    return true
end # isvalid()


# ------------------------------------------------------------------------------
"""
    isvalid(v::NodeValue{d})::Bool where d

Check if the given `d`-dim node value is valid.

## Rules
- 
"""
function isvalid(v::NodeValue{d})::Bool where d
    if isnan(v.f) | isnan(v.α)
        return false
    else
        return true
    end
end # isvalid()


# ------------------------------------------------------------------------------
"""
    nodecmp(node1::Node{d}, node2::Node{d})::Bool where d

Check if the given `d`-dim nodes are equal. This is value comparison which avoid
any reference comparison.
"""
function nodecmp(node1::Node{d}, node2::Node{d})::Bool where d
    for j in 1:d
        if node1.ls[j] != node2.ls[j]; return false; end
        if node1.is[j] != node2.is[j]; return false; end
    end
end # nodecmp()


# ------------------------------------------------------------------------------
"""
    nodecmp_except(node1::Node{d}, node2::Node{d}, j::Int)::Bool where d

Check if the given `d`-dim nodes are equal except the `j`-th dimension.
"""
function nodecmp_except(node1::Node{d}, node2::Node{d}, j::Int)::Bool where d
    for k in 1:d
        if k == j; continue; end
        if (node1.ls[k] != node2.ls[k]) | (node1.is[k] != node2.is[k])
            return false
        end
    end
    return true
end # nodecmp_except()


# ------------------------------------------------------------------------------
"""
    nodecmp_along(node1::Node{d}, node2::Node{d}, j::Int)::Bool where d

Check if the given `d`-dim nodes are equal along the `j`-th dimension.
"""
function nodecmp_along(node1::Node{d}, node2::Node{d}, j::Int)::Bool where d
    if (node1.ls[j] != node2.ls[j]) | (node1.is[j] != node2.is[j])
        return false
    end
    return true
end # nodecmp_along()


# ------------------------------------------------------------------------------
"""
    get_x(l::Int, i::Int)::Float64

Return the one-dimensional hierarchical grid node value for a given level-index
pair `(l, i)`.

## Notes
- given `l`, `i` = 1,2,...,2^(l-1)-1
- private function
- no boundary checks
- no error handling
- error thrown for invalid level-index pair
"""
function get_x(l::Int, i::Int)::Float64
    if (l > 2) & isodd(i)
        return Float64(i / power2(l - 1))
    elseif (l == 2) & (i == 0)
        return 0.0
    elseif (l == 2) & (i == 2)
        return 1.0
    elseif (l == 1) & (i == 1)
        return 0.5
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
end


# ------------------------------------------------------------------------------
"""
    get_x(node::Node{d})::SVector{d, Float64} where d

Return the hierarchical grid point value for a given `d`-dim node.

## Notes
* we use static vectors for assuming not-too-large dimensions (<=100 usually)
"""
function get_x(node::Node{d})::SVector{d, Float64} where d
    return SVector{d, Float64}(get_x.(node.ls, node.is)...)
end


# ------------------------------------------------------------------------------
"""
    fraction2li(i2j::Rational)::NTuple{2,Int}

Convert the given 1-dim node realization denoted as a fraction `i2j` to its lev-
el-index pair `(l,i)`.

## Notes
- Julia's `Rational` does automatically fraction reduction
- This function is useful in finding the neighbor nodes of a specific depth
- This function is useful in listing all the nodes until a specific depth
"""
function fraction2li(i2j::Rational)::NTuple{2,Int}
    if i2j == 0//1
        # case: left boundary
        return (2, 0)
    elseif i2j == 1//1
        # case: right boundary
        return (2, 2)
    elseif i2j == 1//2
        # case: root point
        return (1, 1)
    else
        # case: interior point of level > 2
        return (invpower2(i2j.den) + 1, i2j.num)
    end
end


# ------------------------------------------------------------------------------
"""
    perturb(node::Node{d}, dims::Int, l::Int, i::Int)::Node{d}

Return a new `d`-dim node by perturbing the given node along the `dims` dimensi-
on with the new level `l` and index `i`.
"""
function perturb(node::Node{d}, dims::Int, l::Int, i::Int)::Node{d} where d
    newL = (node.ls[1:dims-1]..., l, node.ls[dims+1:end]...)
    newI = (node.is[1:dims-1]..., i, node.is[dims+1:end]...)
    return Node{d}(newL, newI)
end # perturb()


# ------------------------------------------------------------------------------
"""
    sortperm(nodes2sort::AbstractVector{Node{d}})

Returns the permutation vector that sorts the given `d`-dim nodes in the ascend-
ing order of the (depth, levels by dim, indices by dim) tuple, lexicographically
"""
function Base.sortperm(nodes2sort::AbstractVector{Node{d}}) where d
    return sortperm(NTuple{1+2*d,Int}[
        (node.depth, node.ls..., node.is...)
        for node in nodes2sort
    ])
end # sortperm()


# ------------------------------------------------------------------------------
"""
    sort(nodes2sort::AbstractVector{Node{d}})

Return the sorted `d`-dim nodes in the ascending order of the (depth, levels by
dim, indices by dim) tuple, lexicographically.
"""
function Base.sort(nodes2sort::AbstractVector{Node{d}}) where d
    return nodes2sort[sortperm(nodes2sort)]
end # sort()


# ------------------------------------------------------------------------------
"""
    sort!(nodes2sort::AbstractVector{Node{d}})

Sort the given `d`-dim nodes in the ascending order of the (depth, levels by dim
, indices by dim) tuple, lexicographically.

Make sure the `setindex!()` is defined for `typeof(nodes2sort)`.
"""
function Base.sort!(nodes2sort::AbstractVector{Node{d}}) where d
    nodes2sort[:] .= nodes2sort[sortperm(nodes2sort)]
    return nothing
end # sort()


# ------------------------------------------------------------------------------
"""
    get_child_left(node::Node{d}, dims::Int)::Node{d}

Return the left child node of the given `d`-dim node along `dims` dimension.

## Notes
* when (l == 2) && (i == 0), the `i` of the left child is -1 which is invalid
* when (l == 2) && (i == 2), the `i` of the left child is 3 which is invalid
* does not throw an error if (l == 2) and (i != 0 or 2). But the returned node
is invalid and can be checked by `isvalid()`.
"""
function get_child_left(node::Node{d}, dims::Int)::Node{d} where d
    l = node.ls[dims]
    i = node.is[dims]
    if l > 2
        inew = 2 * i - 1
    elseif (l == 2) && (i == 0)
        inew = -1 # invalid index, will be used to exclude the node
    elseif (l == 2) && (i == 2)
        inew = 3
    elseif l == 1
        inew = 0
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
    return perturb(node, dims, l + 1, inew)
end


# ------------------------------------------------------------------------------
"""
    get_child_right(node::Node{d}, dims::Int)::Node{d}

Return the right child node of the given `d`-dim node along `dims` dimension.

## Notes
* check the docstring of `get_child_left` for the invalid cases
"""
function get_child_right(node::Node{d}, dims::Int)::Node{d} where d
    l = node.ls[dims]
    i = node.is[dims]
    if l > 2
        inew = 2 * node.is[dims] + 1
    elseif (l == 2) && (i == 0)
        inew = 1
    elseif (l == 2) && (i == 2)
        inew = -1  # invalid index, will be used to exclude the node
    elseif l == 1
        inew = 2
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
    return perturb(node, dims, l + 1, inew)
end


# ------------------------------------------------------------------------------
"""
    get_parent(node::Node{d}, dims::Int)::Node{d}

Return the parent node of the given `d`-dim node along `dims` dimension.

## Notes
* special case: l == 3: (3,1) -> (2,0), (3,3) -> (2,2)
* special case: l == 2: (2,0) -> (1,1), (2,2) -> (1,1)
* special case: l == 1: (1,0) -> (0,0)
"""
function get_parent(node::Node{d}, dims::Int)::Node{d} where d
    l = node.ls[dims]
    i = node.is[dims]
    if l > 3
        inew = div(i, 4) * 2 + 1
    elseif (l == 3) && (i == 1)
        inew = 0
    elseif (l == 3) && (i == 3)
        inew = 2
    elseif (l == 2) && (i == 0)
        inew = 1
    elseif (l == 2) && (i == 2)
        inew = 1
    elseif l == 1
        inew = 0
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
    return perturb(node, dims, l - 1, inew)
end


# ------------------------------------------------------------------------------
"""
    get_ghost_left(node::Node{d}, dims::Int, l0::Int)::Node{d}

Return the left nodal point of the given `d`-dim node along `dims` dimension in 
a `l0`-level full hierarchical grid (nodal grid).

## Notes
- This function can be used to find the "ghost" node in the finite difference ap
  plications.
- The left nodal point is the closest left nodal point in the full hierarchical 
  grid of level `l0`.
- Check the source code for the detailed explanation of the algorithm. 
- If there is no lefter nodal point defined, this function returned an invalid
node with `ret.ls[dims]==0` and `ret.is[dims]==-1`.
- We assume the input `node` is a valid node
"""
function get_ghost_left(node::Node{d}, dims::Int, l0::Int)::Node{d} where d
    # the one-dim node splited from the `dims`-dim of `node`
    lj = node.ls[dims]; ij = node.is[dims]

    # discussion by case: l0 --> lj --> ij, nested
    if l0 < 1
        throw(ArgumentError("invalid level $l0"))
    elseif l0 == 1
        # only the root node exist, no left point in any case
        lnew = 0; inew = -1
    elseif l0 == 2
        if (lj == 2) & (ij == 0)
            # the left boundary point, no lefter point
            lnew = 0; inew = -1
        elseif (lj == 2) & (ij == 2)
            # the right boundary point, the left point is the root
            lnew = 1; inew = 1
        elseif (lj == 1) & (ij == 1)
            # the root (center) point, its left point is the left boundary
            lnew = 2; inew = 0
        else
            # invalid points
            lnew = 0; inew = -1
        end
    elseif l0 > 2
        if lj < 1
            throw(ArgumentError("lj<1, invalid input node"))
        elseif lj == 1
            if ij == 1
                # the root point, whose definition is different from lj>2
                # denote the lefter point as a fraction of level `l0`
                _lp = (power2(l0 - 2) - 1) // power2(l0 - 1)
                # then, convert it to the level-index pair
                lnew, inew = fraction2li(_lp)
            else
                throw(ArgumentError("invalid input node"))
            end
        elseif lj == 2
            if ij == 0
                # the left boundary point, no lefter point
                lnew = 0; inew = -1
            elseif ij == 2
                # the right boundary point, the lefter point is the largest inte
                # ior point of level l0
                lnew = l0; inew = power2(l0 - 1) - 1
            else
                throw(ArgumentError("invalid input node"))
            end
        elseif 2 < lj <= l0
            _lp = (ij * power2(l0 - lj) - 1) // power2(l0 - 1)
            lnew, inew = fraction2li(_lp)
        else
            throw(ArgumentError("lj > l0 found, no grid defined"))
        end
    end # if l0

    return perturb(node, dims, lnew, inew)
end


# ------------------------------------------------------------------------------
"""
    get_ghost_right(node::Node{d}, dims::Int, l0::Int)::Node{d}

Return the right nodal point of the given `d`-dim node along `dims` dimension in
a `l0`-level full hierarchical grid (nodal grid).

## Notes
- This function can be used to find the "ghost" node in the finite difference ap
  plications.
- The right nodal point is the nearest right nodal point in the full hierarchicl
grid of level `l0`.
"""
function get_ghost_right(node::Node{d}, dims::Int, l0::Int)::Node{d} where d
    # the one-dim node splited from the `dims`-dim of `node`
    lj = node.ls[dims]; ij = node.is[dims]

    # discussion by case: l0 --> lj --> ij, nested
    if l0 < 1
        throw(ArgumentError("invalid level $l0"))
    elseif l0 == 1
        # only the root node exist, no righter point in any case
        lnew = 0; inew = -1
    elseif l0 == 2
        if (lj == 2) & (ij == 0)
            # the left boundary point, its righter point is the root
            lnew = 1; inew = 1
        elseif (lj == 2) & (ij == 2)
            # the right boundary point, no righter point
            lnew = 0; inew = -1
        elseif (lj == 1) & (ij == 1)
            # the root (center) point, its righter point is the right boundary
            lnew = 2; inew = 2
        else
            # invalid points
            lnew = 0; inew = -1
        end
    elseif l0 > 2
        if lj < 1
            throw(ArgumentError("lj<1, invalid input node"))
        elseif lj == 1
            if ij == 1
                # the root point, whose definition is different from lj>2
                # denote the righter point as a fraction of level `l0`
                _lp = (power2(l0 - 2) + 1) // power2(l0 - 1)
                # then, convert it to the level-index pair
                lnew, inew = fraction2li(_lp)
            else
                throw(ArgumentError("invalid input node"))
            end
        elseif lj == 2
            if ij == 0
                # the left boundary point, the righter point is the 1st interior
                # point of level l0
                lnew = l0; inew = 1
            elseif ij == 2
                # the right boundary point, no righter point
                lnew = 0; inew = - 1
            else
                throw(ArgumentError("invalid input node"))
            end
        elseif 2 < lj <= l0
            _lp = (ij * power2(l0 - lj) + 1) // power2(l0 - 1)
            lnew, inew = fraction2li(_lp)
        else
            throw(ArgumentError("lj > l0 found, no grid defined"))
        end
    end # if l0

    return perturb(node, dims, lnew, inew)
end


# ------------------------------------------------------------------------------
"""
    get_boundaryflag(node::Node{d}, dims::Int)::Int

Return the boundary flag of the given `d`-dim node along `dims` dimension.

## Notes
- the boundary flag is -1 for the left boundary, 0 for the interior, and 1 for t
he right boundary
"""
function get_boundaryflag(node::Node{d}, dims::Int)::Int where d
    if node.ls[dims] == 2
        if node.is[dims] == 0
            return -1
        elseif node.is[dims] == 2
            return 1
        else
            throw(ArgumentError("invalid level-index pair (2, $node.is[dims])"))
        end
    else
        return 0
    end
end


# ------------------------------------------------------------------------------
"""
    get_boundaryflag(node::Node{d})::NTuple{d, Int} where d

Return the boundary flags of the given `d`-dim node along each dimension. Check
`get_boundaryflag(node::Node{d}, dims::Int)` for the definition of the boundary 
flags.
"""
function get_boundaryflag(node::Node{d})::NTuple{d, Int} where d
    return ntuple(j -> get_boundaryflag(node, j), d)
end


# ------------------------------------------------------------------------------
"""
    get_all_1d_nodal_nodes_in_order(l::Int)::Vector{Node{1}}

Return all the 1-dim nodal nodes in the full hierarchical grid of level `l` in
the ascending order of the node value.

## Notes
- This function is used for generating the full nodal grid for testing or other
purposes.
"""
function get_all_1d_nodal_nodes_in_order(l::Int)::Vector{Node{1}}
    if l < 1
        throw(ArgumentError("invalid level $l"))
    elseif l == 1
        return [Node{1}((1,), (1,))]
    elseif l == 2
        return [
            Node{1}((2,), (0,)),
            Node{1}((1,), (1,)),
            Node{1}((2,), (2,))
        ]
    else
        res = Node{1}[]
        den = power2(l - 1)
        for i in 0:power2(l-1)
            lnew, inew = fraction2li(i // den)
            push!(res, Node{1}((lnew,), (inew,)))
        end
        return res
    end
end # get_all_1d_nodal_nodes_in_order()


# ------------------------------------------------------------------------------
"""
    get_dist_along(node1::Node{d}, node2::Node{d}, j::Int)::Float64 where d

Return the distance along the `j`-th dimension between the given `d`-dim nodes.
Order matters: `node1` is the starting point and `node2` is the ending point.
Such that: `node2 - node1`.
"""
function get_dist_along(node1::Node{d}, node2::Node{d}, j::Int)::Float64 where d
    return get_x(node2.ls[j], node2.is[j]) - get_x(node1.ls[j], node1.is[j])
end


# ------------------------------------------------------------------------------
"""
    get_all_1d_regular_index(l::Int)::Vector{Int}

Return all the indices of the regular sparse nodes of level `l`.

## Notes
- If `l == 1`, then only the root node (index 1) is returned.
- If `l == 2`, then only the two boundary nodes (index 0 and 2) are returned.
- If `l > 2`, then all the odd indices from 1 to 2^(l-1)-1 are returned.
"""
function get_all_1d_regular_index(l::Int)::Vector{Int}
    if l > 2
        return collect(1:2:(power2(l-1) - 1))
    elseif l == 2
        return Int[0, 2]
    elseif l == 1
        return Int[1,]
    else
        throw(ArgumentError("invalid level $l"))
    end
end # get_all_1d_regular_index()


# ------------------------------------------------------------------------------
"""
    glue(nodes::Node...)::Node

Return a new node by gluing the given nodes by dimension. The new node's dimens-
ion is the sum of the input nodes' dimensions. The new node's level-index pairs
are the concatenation of the input nodes' level-index pairs.

This function is used to rasing the dimension of the nodes.

For example, let `p1 = Node{1}((1,),(1,))` be the midpoint 0.5 of the 1st dimen-
ion; let `p2 = Node{1}((2,),(0,))` be the left boundary point of the 2nd dimens-
ion. Then, `glue(p1, p2)` returns a new node `Node{2}((1,2),(1,0))` which is the
midpoint of the "bottom" edge of the 2-dim hypercube.
"""
function glue(nodes::Node...)::Node
    lvls = Int[]
    idxs = Int[]
    d    = 0
    for p in nodes
        d += typeof(p).parameters[1]
        append!(lvls, p.ls)
        append!(idxs, p.is)
    end
    return Node{d}(lvls |> NTuple{d,Int}, idxs |> NTuple{d,Int})
end # glue()