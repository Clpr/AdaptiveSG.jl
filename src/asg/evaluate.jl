export evaluate


# ------------------------------------------------------------------------------
"""
    evaluate(
        G::AdaptiveSparseGrid{d},
        x::T ;
        untildepth::Int = 0,
        validonly::Bool = false
    )::Float64 where {d, T<:AbstractVector{Float64}}

Evaluate the multi-linear ASG interpolant at a point `x` which is in the hyper-
cube `[0,1]^d`. Returns a scalar value.

## Arguments
- `G::AdaptiveSparseGrid{d}`: the adaptive sparse grid
- `x::T`: the point to evaluate, where `T` is a vector-like type
- `untildepth::Int`: evaluate only up to this depth (inclusive). Any value less
than 1 means evaluating all nodes.
- `validonly::Bool`: evaluate only the valid nodes. If `true`, then the function
ignores all invalid nodes and nodes with invalid node values.

## Notes
- `x` must be in the hyper-cube `[0,1]^d`. If you need normalization, please use
`normalize()` before calling this function.
- Every evaluation means traversing all the nodes in the grid. The algorithm is
naturally `O(N)` where `N` is the number of nodes in the grid.
- This function is not parallelized due to the difficulty of parallelizing the
traversal of the ordered hash table.
- I understand that `x` can be `NTuple{d, Float64}` in some use cases. However,
please convert it to a vector-like before calling this function for consistency.
- If `d==1`, then use something like `Float64[0.5,]`
"""
function evaluate(
    G::AdaptiveSparseGrid{d},
    x::T ;
    untildepth::Int = 0,
    validonly::Bool = false
)::Float64 where {d, T<:AbstractVector{Float64}}
    # check length once, then use ϕ_unsafe() for performance
    if length(x) != d; throw(ArgumentError("length(x) is not equal to d")); end

    if (untildepth < 1) && (!validonly)
        return _private_evaluate_allnodes_alldepth(G, x)
    elseif (untildepth < 1) && validonly
        return _private_evaluate_validnodes_alldepth(G, x)
    elseif (untildepth >= 1) && (!validonly)
        return _private_evaluate_allnodes_untildepth(G, x, untildepth)
    elseif (untildepth >= 1) && validonly
        return _private_evaluate_validnodes_untildepth(G, x, untildepth)
    else
        throw(ArgumentError("invalid combination of untildepth and validonly"))
    end
end # evaluate()


# ------------------------------------------------------------------------------
"""
    evaluate(
        G::AdaptiveSparseGrid{d},
        node::Node{d} ;
        untildepth::Int = 0,
        validonly::Bool = false
    )::Float64 where {d, T<:AbstractVector{Float64}}

Evaluate the multi-linear ASG interpolant at a node `node`. Returns a `Float64`.
"""
function evaluate(
    G::AdaptiveSparseGrid{d},
    node::Node{d} ;
    untildepth::Int = 0,
    validonly::Bool = false
)::Float64 where {d}
    return evaluate(
        G, get_x(node),
        untildepth = untildepth,
        validonly = validonly
    )
end # evaluate()

# ------------------------------------------------------------------------------
function _private_evaluate_allnodes_alldepth(
    G::AdaptiveSparseGrid{d},
    x::T
)::Float64 where {d, T<:AbstractVector{Float64}}
    fx = 0.0
    for (node, nval) in pairs(G.nv)
        fx += nval.α * ϕ_unsafe(x, node)
    end
    return fx
end
# ------------------------------------------------------------------------------
function _private_evaluate_allnodes_untildepth(
    G::AdaptiveSparseGrid{d},
    x::T,
    depth::Int
)::Float64 where {d, T<:AbstractVector{Float64}}
    fx = 0.0
    for (node, nval) in pairs(G.nv)
        if node.depth <= depth
            fx += nval.α * ϕ_unsafe(x, node)
        end
    end
    return fx
end
# ------------------------------------------------------------------------------
function _private_evaluate_validnodes_alldepth(
    G::AdaptiveSparseGrid{d},
    x::T
)::Float64 where {d, T<:AbstractVector{Float64}}
    fx = 0.0
    for (node, nval) in pairs(G.nv)
        if isvalid(nval) && isvalid(node)
            fx += nval.α * ϕ_unsafe(x, node)
        end
    end
    return fx
end
# ------------------------------------------------------------------------------
function _private_evaluate_validnodes_untildepth(
    G::AdaptiveSparseGrid{d},
    x::T,
    depth::Int
)::Float64 where {d, T<:AbstractVector{Float64}}
    fx = 0.0
    for (node, nval) in pairs(G.nv)
        if isvalid(nval) && isvalid(node) && (node.depth <= depth)
            fx += nval.α * ϕ_unsafe(x, node)
        end
    end
    return fx
end
