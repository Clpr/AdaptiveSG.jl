#===============================================================================
TRAINING PROCEDURES

- trainer        : regular sparse grid (RSG)
    - `rsg!`: init the tree has a given structure
    - `train!`: the tree has a given structure then trained
- trainer/refiner: adaptive sparse grid (ASG)
    - `adapt!`: the tree is refined and continues to grow on the current grid
===============================================================================#
export rsg!
export train!
export adapt!




# ------------------------------------------------------------------------------
"""
    rsg!(
        G        ::SparseGridInterpolant{D},
        maxdepth ::Int,
        maxlevels::Union{NTuple{D,Int},Int},
    ) where {D}

Fill the `G` with the a regular sparse grid structure with the given `maxdepth`
and `maxlevels` along each dimension.

All existing structure in `G` will be cleared and replaced with the new.

The RSG is anisotropic, i.e., the maximum levels can be different along each
dimension. The `maxlevels` is a tuple of integers indicating the maximum levels
along each dimension; `maxdepth` is the depth of the tree, also the so-called
accuracy/refinement level of the grid. If `maxlevels` is a single integer,
then it is interpreted as the same maximum level along each dimension.

## Notes
- If `maxdepth` is less than 2, then an error is thrown. This ensures that the
grid always touches the domain boundary.
"""
function rsg!(
    G        ::SparseGridInterpolant{D},
    maxdepth ::Int,
    maxlevels::Union{NTuple{D,Int},Int},
) where {D}

    if any(maxlevels .< 1); throw(ArgumentError("maxlevels must be >= 1")); end
    if maxdepth < 2; throw(ArgumentError("maxdepth must be >= 2")); end

    maxlevels2 = isa(maxlevels, NTuple) ? maxlevels : ntuple(_ -> maxlevels, D)

    # filter admissible levels & collect nodes
    levels  = NTuple{D,Int}[]
    indices = NTuple{D,Int}[]
    depths  = Int[]

    for ks in Iterators.product([1:l for l in maxlevels2]...)
        if sum(ks) <= (maxdepth + D - 1)
            # given the levels by dimension, collect node indices of the levels
            for Is in Iterators.product([allindices_of_level(k) for k in ks]...)
                push!(levels, ks)
                push!(indices, Is)
                push!(depths, ks |> nodedepth)
            end 
        end
    end

    # modify the grid structure
    G.levels  = stack(levels, dims = 1)
    G.indices = stack(indices, dims = 1)
    G.depths  = depths


    # expand the `fvals`, `coefs` as well to the correct size
    G.fvals = zeros(Float64, levels |> length)
    G.coefs = zeros(Float64, levels |> length)

    return nothing
end # _rsg
# ------------------------------------------------------------------------------
"""
    train!(
        G        ::SparseGridInterpolant{D},
        f        ::Function,
        maxdepth ::Int,
        maxlevels::Union{NTuple{D,Int},Int} ;
        verbose::Bool = false,
    )

Train the sparse grid interpolant `G` with the given function `f` using a
regular sparse grid (RSG) structure with the given `maxdepth` and `maxlevels`
along each dimension.

The target function `f` to fit should be a function that takes a vector of
dimension `D` as the ONLY argument and returns a SINGLE scalar value, which
just matches the definition `f: R^D --> R`. Wrap your function if needed.

The RSG can be anisotropic (i.e., the maximum levels can be different along
each dimension). The `maxlevels` is a tuple of integers indicating the maximum
levels along each dimension; `maxdepth` is the depth of the tree, also the
so-called accuracy/refinement level of the grid. If `maxlevels` is a single
integer, then it is interpreted as the same maximum level along each dimension.
In this case, the grid is isotropic.

The keyword argument `verbose` controls whether to display the training
progress. If `verbose` is `true`, then the training progress will be displayed
in the console.

NOTE: The grid structure in `G` will be overwritten by the new RSG specified by
`maxdepth` and `maxlevels`. If you want to keep the existing grid structure but
simply update the interpolation, use `update!()` instead.
"""
function train!(
    G        ::SparseGridInterpolant{D},
    f        ::Function,
    maxdepth ::Int,
    maxlevels::Union{NTuple{D,Int},Int} ;

    verbose::Bool = false,
) where {D}
    # step: build the RSG structure
    rsg!(G, maxdepth, maxlevels)

    # notes: the `rsg!()` build implies a ascending order of the nodes in `G`
    #        we will use this indexing to update the node values.
    #        this indexing holds because `rsg!()` overwrites the existing
    #        structure in `G`. If the overwriting is removed in the future, then
    #        we need to update this function to handle the existing nodes.

    tic = time()

    verbose && println(
        "Regular sparse grid training starts\n",
        "# dimensions: $D, # total nodes: $(length(G))"
    )

    # update the node values
    for lnow in 1:maxdepth

        # step: filter the nodes at the current level
        idxs   = G.depths .== lnow
        nNodes = sum(idxs)

        verbose && @printf(
            "Training level %d/%d... #nodes = %d, ", 
            lnow, 
            maxdepth,
            nNodes,
        )

        if lnow == 1

            # manually handle the root node (level 1)
            @assert nNodes == 1 "Only 1 root node (level = 1) allowed."
            @assert G.depths[1] == 1 "Root node depth must be in the 1st place."

            x    = get_x(ones(Int, D), ones(Int, D), lb = G.lb, ub = G.ub)
            fval = f(x)
            αval = fval

            G.levels[1, :] = ones(Int, D)  # root node level
            G.indices[1, :] = ones(Int, D) # root node index
            G.fvals[1] = fval
            G.coefs[1] = αval

        else

            # step: update the interpolation coefs for all lnow-level nodes
            for i in findall(idxs)

                x = get_x(
                    G.levels[i, :],
                    G.indices[i, :],
                    lb = G.lb,
                    ub = G.ub,
                )
                fval = f(x)
                αval = fval - _interpolate_until_level(
                    G, x, lnow - 1,
                    safe = false,
                )

                G.fvals[i] = fval
                G.coefs[i] = αval

            end # i

        end # if

        verbose && @printf("Elapsed: %.2f seconds.\n", time() - tic)
    end # lnow

    verbose && @printf("done in %.2f seconds.\n", time() - tic)
    
    return nothing
end # train!
# ------------------------------------------------------------------------------
"""
    adapt!(
        G::SparseGridInterpolant{D},
        f2fit::Function,
        maxdepth::Int ;

        verbose ::Bool    = true,
        tol     ::Float64 = 1e-3,
        toltype ::Symbol  = :absolute, # :absolute or :relative
        parallel::Bool    = false,
    )

Adapt/refine the sparse grid interpolant `G` with the given function `f`
using an adaptive sparse grid (ASG) structure with the given `maxdepth`.

This method will NOT overwrite the existing grid structure in `G`, but
instead refine the existing grid structure by adding new nodes based on the
function `f`. The target function `f` to fit should be a function that takes
a vector of dimension `D` as the ONLY argument and returns a SINGLE scalar
value, which just matches the definition `f: R^D --> R`. Wrap your function
if needed.

The adaption procedure stops when the maximum depth of the grid is reached,
or the tolerance of the hierarchical coefficients is satisfied.

## Notes
- this function does not force `G` to be prepared with an RSG structure but
you may manually prepare the structure and call `update!()` to prepare the
grid structure before calling this function.
- The node growth starts frome the deepest level of the grid, i.e., the
maximum depth of the grid nodes. It will add new nodes of level >= max depth + 1
until the maximum depth of the grid is reached.
- The relative tolerance typically converges faster but may fall into dead loop
at which the function has many zeroes. Default is `false` to use the absolute
tolerance.
- The `parallel` argument controls whether to use parallel processing for the
training process.
"""
function adapt!(
    G       ::SparseGridInterpolant{D},
    f2fit   ::Function,
    maxdepth::Int ;

    verbose ::Bool    = true,
    tol     ::Float64 = 1e-3,
    toltype ::Symbol  = :absolute, # absolute or relative
    parallel::Bool    = false,
) where {D}

    maxdepthNow = maximum(G.depths)

    @assert length(G) > 0 "The grid structure in G is empty."
    @assert maxdepthNow <= maxdepth "maxdepth must be >= current depth."
    @assert maxdepthNow > 0 "Ill-defined grid structure: maxdepth < 1 found."

    atol, rtol = if toltype == :absolute
        tol, NaN
    elseif toltype == :relative
        NaN, tol
    else
        throw(ArgumentError("toltype must be :absolute or :relative."))
    end

    # current parent depth
    lnow = maxdepthNow


    verbose && println(
        "Adaptive sparse grid adaption starts\n",
        "# dimensions: $D, # current total nodes: $(length(G)), ",
        toltype, " tolerance: ", tol,
    )
    tic = time()

    if lnow >= maxdepth
        verbose && @printf(
            "Current depth %d >= max depth %d, stop adaption.\n", 
            lnow, maxdepth
        )
        return nothing
    end

    # adaption
    while lnow < maxdepth

        # step: locate the row index of all the parent nodes to grow at
        irowParents::Vector{Int} = findall(G.depths .== lnow)

        verbose && @printf(
            "level %d/%d, #parents: %d, ",
            lnow,
            maxdepth,
            length(irowParents)
        )
        

        #=----------------------------------------------------------------------
        NOTES

        - Each parent node can grow 2*D children nodes (2 children for each dim)
          For total N parent nodes, there are at most 2*D*N children nodes.

        - In multi-dimensional case, some children nodes may be overlapping or
          in other words, duplicate/identical. This is because the node tree
          is not necessarily full

        - Some grown children nodes may not be valid, i.e., they are out of
          the domain of the grid. This happens when the parent node is at the
          boundary (of one or more dimensions) of the grid domain. At such dims,
          only the "inward" children node is valid out of the two children along
          that dimension.

        - Performance bottle necks
            - Find, filter all possible children nodes at the current level.
            - Evaluation of the function at the children nodes.
            - Evaluation of the interpolant until the current (parent) level.

        - However, prepare too many candidate children nodes and then filter
          them is not efficient, especially when the number of parent nodes is
          large.

        - Thus, 1 parallel loops are used:
            1. parallelize the children node finding
            2. parallelize the function evaluation at the children nodes
        
        - After the loop, the overlapped children nodes are filtered:
            - only one copy is kept


        ----------------------------------------------------------------------=#

        thdres = [
            Dict{NTuple{2,Vector{Int}},NTuple{2,Float64}}()
            for _ in 1:Threads.nthreads()
        ]
            
        @maybe_threads parallel for iParent in irowParents

            tid = Threads.threadid()

            lvlParent = G.levels[iParent, :]
            idxParent = G.indices[iParent, :]

            for j in 1:D, side in (:left, :right)

                # step: grow to the next level, along dimension `j`
                lvlChild, idxChild = child(lvlParent, idxParent, j, side = side)

                # step: if the children node is valid, then do residual fitting
                if isvalid(lvlChild, idxChild)

                    x    = get_x(lvlChild, idxChild, lb = G.lb, ub = G.ub)
                    fval = f2fit(x)
                    αval = fval - G(x, safe = false, extrapolate = false)

                    if isbig2add(αval,fval,rtol,atol,toltype == :relative)

                        _key = (lvlChild, idxChild)
                        _val = (fval, αval)

                        if !haskey(thdres[tid], _key)
                            thdres[tid][_key] = _val
                        end

                    end # if

                end # if

            end # j, side

        end # iParent


        # step: early stop if no new children nodes are found
        if all(length.(thdres) .== 0)
            verbose && @printf(
                "\nConverged: no new nodes found. Elapsed: %.2f seconds\n",
                time() - tic
            )
            break
        end

        # step: merge all the thread-specific dictionaries
        # NOTES: `merge` automatically removes the duplicated children nodes
        newNodes = reduce(merge, thdres)

        # step: stack the children nodes
        nChildren = length(newNodes)
        lvlChild  = Matrix{Int}(undef, nChildren, D)
        idxChild  = Matrix{Int}(undef, nChildren, D)
        depChild  = Vector{Int}(undef, nChildren)
        fChild    = Vector{Float64}(undef, nChildren)
        αChild    = Vector{Float64}(undef, nChildren)

        for (i, ((Ls,Is),(fval,αval))) in enumerate(newNodes)

            lvlChild[i, :] = Ls
            idxChild[i, :] = Is
            depChild[i]    = nodedepth(Ls)
            fChild[i]      = fval
            αChild[i]      = αval

        end # i

        
        # step: update the grid structure with the new children nodes
        G.levels  = vcat(G.levels , lvlChild)
        G.indices = vcat(G.indices, idxChild)
        G.depths  = vcat(G.depths , depChild)
        G.fvals   = vcat(G.fvals  , fChild)
        G.coefs   = vcat(G.coefs  , αChild)

        
        verbose && @printf(
            "#added/#candidates/#total: %d/%d/%d, Elapsed: %.2f seconds.\n",
            length(newNodes),
            length(irowParents) * 2 * D,
            length(G) + length(newNodes),
            time() - tic,
        )


        # move to the next level
        lnow += 1

        if lnow == maxdepth
            verbose && @printf(
                "Reached the maximum depth %d. Elapsed: %.2f seconds.\n", 
                maxdepth, time() - tic
            )
            break
        end
    end # lnow

    return nothing
end # adapt!













