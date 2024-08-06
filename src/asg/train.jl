export train!


# ------------------------------------------------------------------------------
"""
    train!(
        G::AdaptiveSparseGrid{d}, 
        f2fit::Function ;
        printlevel::String = "iter"
    ) where d

Train the adaptive sparse grid `G` by adding new nodes based on the function 
`f2fit` until the maximum depth `max_depth` is reached.

## Arguments
- `G::AdaptiveSparseGrid{d}`: the adaptive sparse grid to train
- `f2fit::Function`: the function to fit, which must return a single `Float64`,
while being able to accept a `SVector{d, Float64}` as the only position argument
check `isvalid_f2fit()` for more details
- `printlevel::String`: print level, either "iter", "final", or "none"

## Notes
- The training proecss is initialized by adding all depth 1, 2 nodes. The max
depth must be at least 3. This validation happened in the constructor.
- The `f2fit` "must be able to" means it can be defined in a more generic way
such as `f2fit(x::AbstractVector{Float64})` or `f2fit(x)` without type assertion
"""
function train!(
    G::AdaptiveSparseGrid{d}, 
    f2fit::Function ;
    printlevel::String = "iter"
) where d

    # check the function-to-fit's input and output types
    validate_f2fit!(f2fit, d)

    # if the grid is already initialized, then the current algorithm does not
    # clear and re-initialize the grid to avoid potential data loss.
    if length(G) > 0
        throw(ArgumentError(string(
            "non-empty grid. Data loss may occur. ",
            "If you want to re-train the entire ASG ",
            "without changing the grid structure, ",
            "consider `update_all!()` instead",
            "; otherwise, create a new AdaptiveSparseGrid{d}."
        )))
    end

    # initialize the grid by adding all level 1 (center) & 2 (boundary) nodes
    begin

        # node: depth 1, unique root for all dimensions: ([1,1,...], [1,1,...])
        p11::Node{d} = Node{d}(ntuple(x -> 1, d), ntuple(x -> 1, d))
        f11::Float64 = p11 |> get_x |> f2fit
        insert!(G.nv, p11, NodeValue{d}(
            f11, # function value, i.e. nodal coefficient
            f11, # hierarchical coefficient, = nodal coef at the root
        ))

        # loop over all dimensions for level 2 points
        # Tips: any growth from (1,1,...) cannot have out-of-boundary children, 
        #       thus there is no validation check for the children
        for j in 1:d
            # depth 2: left boundary point: ls[j] = 2, is[j] = 0
            p20 = get_child_left(p11, j)
            x20 = get_x(p20) ::SVector{d, Float64}
            f20 = x20 |> f2fit
            α20 = f20 - f11 * ϕ_unsafe(x20, p20) # residual fitting
            insert!(G.nv, p20, NodeValue{d}(f20, α20))
    
            # depth 2: right boundary point ls[j] = 2, is[j] = 2
            p22 = get_child_right(p11, j)
            x22 = get_x(p22) ::SVector{d, Float64}
            f22 = x22 |> f2fit
            α22 = f22 - f11 * ϕ_unsafe(x22, p22)
            insert!(G.nv, p22, NodeValue{d}(f22, α22))
        end # j

        # set depth to 2
        G.depth = 2
    end # begin


    # loop until the `max_depth` is reached, start growth from level 2
    #
    # Tips: `lnow` indicates the current depth of the tree in which we are searc
    #       hing for children of the nodes of depth `lnow`
    #
    # Tips: we never drop nodes with depth <= 2 regardless of `rtol`
    #
    # Tips: easy to know that, for any `lnow` >= 2, the existing nodes must be
    #       "important" to stay.

    lnow = 2 # the current parent depth

    while lnow < G.max_depth
        # Tips: for thread safety, we use thread-specific arrays to save the new
        #       nodes as vectors of `Pair{Node{d}, NTuple{2,Float64}}`. After th
        #       e loop, we insert the new nodes into the hash table.
        # 
        # Tips: such trail-collect-insert strategy allows us to safely evaluate
        #       the whole interpolant for all candidate children nodes before
        #       inserting them into the hash table.

        newnodes = [
            Pair{Node{d}, NodeValue{d}}[] 
            for _ in 1:Threads.nthreads()
        ]

        # Tips: parallelize the training of level `lnow` for all candidate nodes
        #       whose number is `(2^lnow)^d` in the worst case
        #
        # Tips: we assume there are always enough RAM to storage the pre-collect
        #       ed nodes. The pre-collection is for @threads. On a PC with 16GB 
        #       RAM, too-high dimension (e.g. > 100) is not recommended.
        #
        # Tips: because we loop over the parent depth and only push new nodes af
        #       ter finding all of them, so it is safe to evaluate the current
        #       interpolant until the current depth.
        #
        # Tips: the rule of adding new nodes is scaled by the function value at
        #       the candidate child node.
        # 
        # Tips: I expect `candidate_parent_nodes` to be not too large, because
        #       if this is true, then the number of candidate children nodes is
        #       `(2^lnow)^d` which is infeasible.

        candidate_parent_nodes = Iterators.Filter(
            node -> node.depth == lnow,
            keys(G.nv)
        ) |> collect

        # Tips: the outer loop is dimension which is relatively much smaller th-
        #       an the number of candidate nodes. Thus, we parallelize the inner
        #       loop in which the number of candidate nodes grows exponentially
        #       as the depth increases.
        #
        # Tips: the built-in `Threads.@threads` does not support nested loop, so
        #       we have to use a workaround to parallelize the inner loop. If in
        #       a future version of Julia this is supported, then it is definite
        #       better to @threads the whole nested loop and let the compiler to
        #       allocate the workloads to threads.
        #
        # Tips: all operations in the inner loop is thread-safe by:
        #       1. the only in-place operation is `push!` which is thread-safe
        #       2. all the other operations are read-only
        #       in the future optimization, be careful with the thread-safety

        for j in 1:d
            Threads.@threads for pnode in candidate_parent_nodes

                # obtain the current thread's id to know where to save results
                tid = Threads.threadid()

                # get left & right children of `pnode` along dimension `j`
                pl = get_child_left(pnode, j) 
                pr = get_child_right(pnode, j)

                # if valid, then do residual fitting and decide if to cache it
                if isvalid(pl)
                    x = get_x(pl)          # x values of the node
                    f = f2fit(x)           # nodal coefficient
                    α = f - evaluate(G, x) # hierarchical coefficient
                    if isbig2add(α, f, G.rtol)
                        push!(newnodes[tid], pl => NodeValue{d}(f, α))
                    end
                end # if
                if isvalid(pr)
                    x = get_x(pr)
                    f = f2fit(x)
                    α = f - evaluate(G, x)
                    if isbig2add(α, f, G.rtol)
                        push!(newnodes[tid], pr => NodeValue{d}(f, α))
                    end
                end # if

            end # pnode
        end # j

        # update the hash table with the new nodes
        if all(length.(newnodes) .== 0)
            if printlevel == "iter"
                println("no new nodes added at depth $(lnow), training stops.")
            end
            break
        else
            _counter_newnodes = 0
            for newnodes_thread in newnodes
                for (node, nval) in newnodes_thread
                    if !haskey(G.nv, node)
                        insert!(G.nv, node, nval)
                        _counter_newnodes += 1
                    end
                end
            end

            if printlevel == "iter"
                _nbr_candidates  = length(candidate_parent_nodes) * 2 * d
                println(
                    "depth ", lnow, "/", G.max_depth, " : ",
                    "new nodes/candidates = ",
                    _counter_newnodes, "/", _nbr_candidates, 
                    ", current #nodes = ", length(G)
                )
            end

            lnow += 1
            continue
        end # if

    end # lnow

    # update the current depth
    G.depth = lnow

    if (printlevel == "iter") || (printlevel == "final")
        if G.depth < G.max_depth
            println(
                "Converged at depth ", lnow, " < max depth ", G.max_depth,
                "; #nodes = ", length(G)
            )
        else
            println(
                "Algorithm prematured at depth ", lnow, 
                " = max depth ", G.max_depth,
                "; #nodes = ", length(G)
            )
        end 
    end
    return nothing
end # train!
