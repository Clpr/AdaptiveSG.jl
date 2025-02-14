export evaluate


# ------------------------------------------------------------------------------
"""
    _evaluate_conditional(
        G::AbstractSparseGrid{d},
        x::AbstractVector{Float64} ;
        untildepth::Int  = 0,
        validonly ::Bool = false
    )::Float64

Evaluate the multi-linear ASG/RSG interpolant at a point `x` which is in the hy-
percube `[0,1]^d`. Returns a scalar value. This function do this evaluation con-
ditionally based on the `untildepth` and `validonly` arguments.

## Arguments
- `G::AbstractSparseGrid{d}`: the adaptive or regular sparse grid
- `x::AbstractVector{Float64}`: the point to evaluate
- `untildepth::Int`: evaluate only up to this depth (inclusive). Any value less
than 1 means evaluating all nodes.
- `validonly::Bool`: evaluate only the valid nodes. If `true`, then the function
ignores all invalid nodes and nodes with invalid node values.

## Notes
- This function is memory unsafe and for internal use. If you do need it, please
be careful with the memory management.
- This function has no normalization version.
"""
function _evaluate_conditional(
    G::AbstractSparseGrid{d},
    x::AbstractVector{Float64} ;
    untildepth::Int  = 0,
    validonly ::Bool = false
)::Float64 where {d}
    fx = 0.0
    stopdepth = (untildepth < 1) ? G.max_depth : untildepth

    if validonly
        for (node, nval) in G.nv |> pairs
            flgvld = isvalid(nval) && isvalid(node)
            if flgvld && (node.depth <= stopdepth)
                fx += nval.α * ϕ_unsafe(x, node)
            end
        end
        return fx
    else
        # to reduce the number of justification times
        for (node, nval) in G.nv |> pairs
            if node.depth <= stopdepth
                fx += nval.α * ϕ_unsafe(x, node)
            end
        end
        return fx
    end
end # _evaluate_conditional()






# ------------------------------------------------------------------------------
"""
    evaluate(
        G::AbstractSparseGrid{d},
        x::AbstractVector{Float64} ;
        extrapolation::Bool = false
    )::Float64

Evaluate the multi-linear ASG/RSG interpolant at a point `x` which is in the hy-
percube `[0,1]^d`. Returns a scalar value.

## Arguments
- `G::AbstractSparseGrid{d}`: the adaptive or regular sparse grid
- `x::AbstractVector{Float64}`: the point to evaluate
- `extrapolation::Bool`: if `true`, then the function calculates the linear
extrapolation of the interpolant at the point `x` if it is outside the domain
`[0,1]^d`. If `false`, then an outside point `x` will return `0.0` due to the 
nature of the compact support of the basis functions.

## Notes
- Every evaluation means traversing all the nodes in the grid. The algorithm is
naturally `O(N)` where `N` is the number of nodes in the grid.
- This function is not parallelized since I expect the number of nodes to be
not too large (sparsity).
- I understand that `x` can be `NTuple{d, Float64}` in some use cases. However,
please convert it to a vector-like before calling this function for consistency.
- If `d==1`, then use something like `Float64[0.5,]`
"""
function evaluate(
    G::AbstractSparseGrid{d},
    x::AbstractVector{Float64} ;
    extrapolation::Bool = false
)::Float64 where {d}
    #=
    NOTES: numerically it is equivalent for an interior `x` to call the `extra-
    polation()`. It has similar time complexity as the bear bones evaluaion. 
    However, the extrapolation() function requires much more allocations and 
    computations. Therefore, it is better to have a separate function for the 
    interior evaluation.
    =#
    if extrapolation
        return extrapolate(G, x)
    else
        # check length once, then use ϕ_unsafe() for performance
        if length(x) != d; throw(ArgumentError("length(x) != d found")); end
        fx = 0.0
        for (node, nval) in pairs(G.nv)
            fx += nval.α * ϕ_unsafe(x, node)
        end
        return fx
    end
end # evalaute



# ------------------------------------------------------------------------------
"""
    evaluate(
        G::AbstractSparseGrid{d},
        x::AbstractVector{Float64},
        nzer::Normalizer{d} ;
        extrapolation::Bool = false
    )::Float64

Evaluates the ASG/RSG interpolant at a point `x` from a hyper-rectangle defined
by `nzer`. This function is a wrapper around `evaluate()` and `normalize()`.
"""
function evaluate(
    G   ::AbstractSparseGrid{d},
    x   ::AbstractVector{Float64},
    nzer::Normalizer{d} ;
    extrapolation::Bool = false
)::Float64 where {d}
    x01 = normalize(x, nzer)
    return evaluate(G, x01, extrapolation = extrapolation)
end # evaluate





# ------------------------------------------------------------------------------
function evaluate(
    G   ::AbstractSparseGrid{d},
    node::Node{d} ;
    extrapolation::Bool = false
)::Float64 where {d}
    return evaluate(G, get_x(node), extrapolation = extrapolation)
end
# ------------------------------------------------------------------------------
function evaluate(
    G   ::AbstractSparseGrid{d},
    node::Node{d},
    nzer::Normalizer{d} ;
    extrapolation::Bool = false
)::Float64 where {d}
    return evaluate(G, get_x(node), nzer, extrapolation = extrapolation)
end

