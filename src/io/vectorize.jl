export vectorize_levels
export vectorize_indices
export vectorize_nodal
export vectorize_hierarchical
export vectorize_x
export vectorize_depth


# ------------------------------------------------------------------------------
"""
    vectorize_levels(G::AdaptiveSparseGrid{d})::Matrix{Int}

Vectorize the levels of the nodes in the adaptive sparse grid `G`. The result is
a matrix where each row corresponds to the level of a node in the grid, and the
columns correspond to the dimensions of the grid in the order of the dimensions.

Returns a `length(G) * d` matrix. For the same `G`, the row order across the 
vectorized quantities is the same (stacking order).
"""
function vectorize_levels(G::AdaptiveSparseGrid{d})::Matrix{Int} where {d}
    res = Matrix{Int}(undef, length(G), d)
    for (i, node) in enumerate(keys(G.nv))
        res[i, :] .= node.ls
    end
    return res
end # matrixize_levels()


# ------------------------------------------------------------------------------
"""
    vectorize_indices(G::AdaptiveSparseGrid{d})::Matrix{Int}

Vectorize the indices of the nodes in the adaptive sparse grid `G`. The result 
is a matrix where each row corresponds to the index of a node in the grid, and
the columns correspond to the dimensions of the grid in the order of the dimens-
ions.

Returns a `length(G) * d` matrix. For the same `G`, the row order across the 
vectorized quantities is the same (stacking order).
"""
function vectorize_indices(G::AdaptiveSparseGrid{d})::Matrix{Int} where {d}
    res = Matrix{Int}(undef, length(G), d)
    for (i, node) in enumerate(keys(G.nv))
        res[i, :] .= node.is
    end
    return res
end # vectorize_indices()


# ------------------------------------------------------------------------------
"""
    vectorize_nodal(G::AdaptiveSparseGrid{d})::Vector{Float64}

Vectorize the nodal coefficient (function value) of the nodes in the adaptive
sparse grid `G`. The result is a vector where each element corresponds to the
function value of a node in the grid.

Returns a `length(G)` vector. For the same `G`, the row order across the 
vectorized quantities is the same (stacking order).
"""
function vectorize_nodal(G::AdaptiveSparseGrid{d})::Vector{Float64} where {d}
    res = Vector{Float64}(undef, length(G))
    for (i, nval) in enumerate(G.nv)
        res[i] = nval.f
    end
    return res
end # vectorize_nodal()


# ------------------------------------------------------------------------------
"""
    vectorize_hierarchical(G::AdaptiveSparseGrid{d})::Vector{Float64} where d

Vectorize the hierarchical coefficient of the nodes in the adaptive sparse grid
`G`. The result is a vector where each element corresponds to the hierarchical
coefficient of a node in the grid.

Returns a `length(G)` vector. For the same `G`, the row order across the
vectorized quantities is the same (stacking order).
"""
function vectorize_hierarchical(
    G::AdaptiveSparseGrid{d}
)::Vector{Float64} where {d}
    res = Vector{Float64}(undef, length(G))
    for (i, nval) in enumerate(G.nv)
        res[i] = nval.α
    end
    return res
end # vectorize_hierarchical()


# ------------------------------------------------------------------------------
"""
    vectorize_x(G::AdaptiveSparseGrid{d})::Matrix{Float64} where d

Vectorize the coordinate values of the nodes in the adaptive sparse grid `G`.
The result is a matrix where each row corresponds to the coordinate values of a
node in the grid, and the columns correspond to the dimensions of the grid in
the order of the dimensions.

Returns a `length(G) * d` matrix. For the same `G`, the row order across the 
vectorized quantities is the same (stacking order).
"""
function vectorize_x(G::AdaptiveSparseGrid{d})::Matrix{Float64} where {d}
    res = Matrix{Float64}(undef, length(G), d)
    for (i, node) in enumerate(keys(G.nv))
        res[i, :] .= get_x(node)
    end
    return res
end # vectorize_x()


# ------------------------------------------------------------------------------
"""
    vectorize_depth(G::AdaptiveSparseGrid{d})::Vector{Int} where d

Vectorize the depth of the nodes in the adaptive sparse grid `G`. The result is
a vector where each element corresponds to the depth of a node in the grid.

Returns a `length(G)` vector. For the same `G`, the row order across the vector-
ized quantities is the same (stacking order).
"""
function vectorize_depth(G::AdaptiveSparseGrid{d})::Vector{Int} where {d}
    res = Vector{Int}(undef, length(G))
    for (i, node) in enumerate(keys(G.nv))
        res[i] = node.depth
    end
    return res
end # vectorize_depth()


