export save!


# ------------------------------------------------------------------------------
"""
    save!(filename::String, G::AdaptiveSparseGrid{d}) where d

Export the adaptive sparse grid `G` to an HDF5 file `filename`.

## Data properties
- `class` [String]: identifier string, "AdaptiveSparseGrid"
- `d` [Int]: dimension of the grid
- `depth` [Int]: depth of the grid
- `max_depth` [Int]: maximum depth of the grid
- `rtol` [Float64]: relative tolerance of the grid
- `selfcontained` [Bool]: whether the grid is self-contained
- `levels` [Matrix{Int}]: stacked levels of the nodes in the grid
- `indices` [Matrix{Int}]: stacked indices of the nodes in the grid
- `nodal` [Matrix{Float64}]: stacked nodal coefficients of the nodes
- `hierarchical` [Matrix{Float64}]: stacked hierarchical coefficients

To load the grid back, use `load(filename, AdaptiveSparseGrid{d})` where the 2nd
argument is a `DataType` but not an instance.
"""
function save!(filename::String, G::AdaptiveSparseGrid{d}) where {d}
    HDF5.h5open(filename, "w") do file
        HDF5.write(file, "class", "AdaptiveSparseGrid")
        HDF5.write(file, "d", d)
        HDF5.write(file, "depth", G.depth)
        HDF5.write(file, "max_depth", G.max_depth)
        HDF5.write(file, "rtol", G.rtol)
        HDF5.write(file, "levels", vectorize_levels(G))
        HDF5.write(file, "indices", vectorize_indices(G))
        HDF5.write(file, "nodal", vectorize_nodal(G))
        HDF5.write(file, "hierarchical", vectorize_hierarchical(G))
    end
    return nothing
end # save!()


# ------------------------------------------------------------------------------
"""
    save!(filename::String, yp::YellowPages{d}) where d

Export the yellow pages `yp` to an HDF5 file `filename`.

## Data properties
- `class` [String]: identifier string, "YellowPages"
- `d` [Int]: dimension of the grid
- `type` [String]: type of the yellow pages, either "sparse" or "ghost"
- `levels` [Matrix{Int}]: stacked levels of the nodes in the yellow pages
- `indices` [Matrix{Int}]: stacked indices of the nodes in the yellow pages
- `left_levels` [Matrix{Int}]: stacked levels of the left children
- `right_levels` [Matrix{Int}]: stacked levels of the right children
- `left_indices` [Matrix{Int}]: stacked indices of the left children
- `right_indices` [Matrix{Int}]: stacked indices of the right children

To load the yellow pages back, use `load(filename, YellowPages{d})`. The 2nd
argument is a `DataType` but not an instance.
"""
function save!(filename::String, yp::YellowPages{d}) where d
    # vectorize the yellow page nodes of `address`
    n = length(yp.address)
    lvls = Matrix{Int}(undef, n, d)
    idxs = Matrix{Int}(undef, n, d)
    for (node, i) in pairs(yp.address)
        lvls[i, :] .= node.ls
        idxs[i, :] .= node.is
    end
    # vectorize the yellow page nodes of `left` and `right`
    left_lvls  = Matrix{Int}(undef, n, d)
    right_lvls = Matrix{Int}(undef, n, d)
    left_idxs  = Matrix{Int}(undef, n, d)
    right_idxs = Matrix{Int}(undef, n, d)
    for i in 1:n, j in 1:d
        left_lvls[i, :]  .= yp.left[i,j].ls
        right_lvls[i, :] .= yp.right[i,j].ls
        left_idxs[i, :]  .= yp.left[i,j].is
        right_idxs[i, :] .= yp.right[i,j].is
    end

    HDF5.h5open(filename, "w") do file
        HDF5.write(file, "class", "YellowPages")
        HDF5.write(file, "d", d)
        HDF5.write(file, "type", string(yp.type))
        HDF5.write(file, "levels", lvls)
        HDF5.write(file, "indices", idxs)
        HDF5.write(file, "left_levels", left_lvls)
        HDF5.write(file, "right_levels", right_lvls)
        HDF5.write(file, "left_indices", left_idxs)
        HDF5.write(file, "right_indices", right_idxs)
    end
    return nothing
end # save!()


