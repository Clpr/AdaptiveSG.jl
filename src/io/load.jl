export load!


# ------------------------------------------------------------------------------
"""
    load!(filename::String, dtype::DataType)

Load the data from an HDF5 file `filename` and return the data as an instance of
the data type `dtype`. The `dtype` must be a `DataType` but not an instance.
The available data types are `AdaptiveSparseGrid` and `YellowPages` or their 
parametric versions.

Notice that this function returns a **new** instance.
"""
function load!(filename::String, dtype::DataType)
    if dtype <: AdaptiveSparseGrid
        return _load_asg!(filename)
    elseif dtype <: YellowPages
        return _load_yp!(filename)
    else
        error("Unknown data type: $dtype")
    end
end # load!()


# # ------------------------------------------------------------------------------
# function _load_asg!(filename::String)::AdaptiveSparseGrid{d} where {d}
    # G = AdaptiveSparseGrid{d}(max_depth, rtol = rtol)
    # HDF5.h5open(filename, "r") do file
    #     # check the class
    #     tyname = HDF5.read(file, "class") ::String
    #     if tyname != "AdaptiveSparseGrid"
    #         error("Class in the file mismatch: $tyname")
    #     end
    #     # get dimensionality
    #     dims = HDF5.read(file, "d") ::Int
    #     if dims != d
    #         error("Dimensionality mismatch: $dims != $d")
    #     end
    #     # get max_depth
    #     max_depth = HDF5.read(file, "max_depth") ::Int
    #     depth = HDF5.read(file, "depth") ::Int
    #     # get tolerance
    #     rtol = HDF5.read(file, "rtol") ::Float64
    #     # get selfcontained
    #     selfcontained = HDF5.read(file, "selfcontained") ::Bool

    #     G.depth = depth

    #     # load levels, indices, nodal, and hierarchical coefficients
    #     lvls = HDF5.read(file, "levels") ::Matrix{Int}
    #     idxs = HDF5.read(file, "indices") ::Matrix{Int}
    #     nodal = HDF5.read(file, "nodal") ::Vector{Float64}
    #     hiera = HDF5.read(file, "hierarchical") ::Vector{Float64}
    #     n = size(lvls, 1)

    #     for i in 1:n
    #         insert!(
    #             G.nv,
    #             Node{d}(
    #                 lvls[i, :] |> Tuple,
    #                 idxs[i, :] |> Tuple
    #             ),
    #             NodeValue{d}(nodal[i], hiera[i])
    #         )
    #     end
    # end
#     return G
# end # _load_asg!()


# ------------------------------------------------------------------------------
function _load_yp!(filename::String)::YellowPages{d} where {d}
    # malloc
    dat_meta = Dict{String, Union{Int, Symbol}}()
    dat      = Dict{String, Union{Matrix{Int}, Matrix{Node{d}}}}()

    HDF5.h5open(filename, "r") do file
        # check the class
        tyname = HDF5.read(file, "class") ::String
        if tyname != "YellowPages"
            error("Class in the file mismatch: $tyname")
        end

        # get dimensionality
        dat_meta["d"] = HDF5.read(file, "d") ::Int
        if dat_meta["d"] != d
            error("Dimensionality mismatch: $(dat_meta["d"]) != $d")
        end

        # get neighbor type
        ntype = Symbol(HDF5.read(file, "type")) ::Symbol
        if ntype != :sparse && ntype != :ghost
            error("Unknown neighbor type: $ntype")
        end
        dat_meta["type"] = ntype

        # get address information
        dat["address_lvls"] = HDF5.read(file, "address_levels")  ::Matrix{Int}
        dat["address_idxs"] = HDF5.read(file, "address_indices") ::Matrix{Int}
        n = size(dat["address_lvls"], 1)

        # get neighbor matricies
        local left_lvls  = HDF5.read(file, "left_levels")   ::Matrix{Int}
        local right_lvls = HDF5.read(file, "right_levels")  ::Matrix{Int}
        local left_idxs  = HDF5.read(file, "left_indices")  ::Matrix{Int}
        local right_idxs = HDF5.read(file, "right_indices") ::Matrix{Int}
        dat["left"]  = Matrix{Node{d}}(undef, n, d)
        dat["right"] = Matrix{Node{d}}(undef, n, d)
        for i in 1:n, j in 1:d
            dat["left"][i,j] = Node{d}(
                left_lvls[i,:] |> Tuple, 
                left_idxs[i,:] |> Tuple
            )
            dat["right"][i,j] = Node{d}(
                right_lvls[i, :] |> Tuple, 
                right_idxs[i, :] |> Tuple
            )
        end
    end

    return YellowPages{d}(
        Dictionary{Node{d}, Int}(
            Node{d}[
                Node{d}(
                    dat["address_lvls"][i, :] |> Tuple, 
                    dat["address_idxs"][i, :] |> Tuple
                )
                for i in 1:size(dat["address_lvls"], 1)
            ],
            1:size(dat["address_lvls"], 1)
        ),  # address
        dat["left"], 
        dat["right"], 
        dat_meta["type"],
    )
end # _load_yp!()








