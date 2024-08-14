export apply
export stencil2jv


# ------------------------------------------------------------------------------
"""
    Base.:(==)(stc1::LinearStencil{d}, stc2::LinearStencil{d})::Bool

Equality comparison of two linear stencils. Two linear stencils are equal if the
weights of the nodes in the two stencils are equal.

Notice that the `weights` field is an ordered dictionary, so the order of the
nodes in the stencil does matter.
"""
function Base.:(==)(stc1::LinearStencil{d},stc2::LinearStencil{d})::Bool where d
    return stc1.weights == stc2.weights
end


# ------------------------------------------------------------------------------
"""
    Base.:+(stc1::LinearStencil{d}, stc2::LinearStencil{d})::LinearStencil{d}

Addition of two linear stencils. The nodes in the new stencil are the union of
the nodes in the two stencils. If a node is in both stencils, then the weight
of the node in the new stencil is the sum of the weights in the two stencils.

This operation is defined for the following operations:

Let stencil 1 be `∑_{q∈L1} w1(q) f(q)` and stencil 2 be `∑_{q∈L2} w2(q) f(q)`.
Then, the new stencil is `∑_{q∈L1∪L2} w(q) f(q)`, where `w(q) = w1(q) + w2(q)`.
"""
function Base.:+(
    stc1::LinearStencil{d}, 
    stc2::LinearStencil{d}
)::LinearStencil{d} where d
    # directly start based on a deep copy of stc1
    neww = deepcopy(stc1.weights)

    # go over: stc2
    for (node, w) in pairs(stc2.weights)
        if haskey(neww, node)
            neww[node] += w
        else
            insert!(neww, node, w)
        end
    end

    return LinearStencil{d}(neww)
end # Base.:+()


# ------------------------------------------------------------------------------
"""
    Base.:+(stc::LinearStencil{d}, a::T)::LinearStencil where {d, T <: Real}

Addition of a scalar `a` and a linear stencil `stc`. The weights of the new ste-
ncil are the sum of the scalar and the weights of the original stencil pointwise
"""
function Base.:+(stc::LinearStencil{d}, a::T)::LinearStencil where {d,T <: Real}
    return LinearStencil{d}(a .+ stc.weights)
end # Base.:+()
function Base.:+(a::T, stc::LinearStencil{d})::LinearStencil where {d,T <: Real}
    return LinearStencil{d}(a .+ stc.weights)
end # Base.:+()


# ------------------------------------------------------------------------------
function Base.:-(
    stc1::LinearStencil{d}, 
    stc2::LinearStencil{d}
)::LinearStencil{d} where d
    neww = deepcopy(stc1.weights)
    for (node, w) in pairs(stc2.weights)
        if haskey(neww, node)
            neww[node] -= w
        else
            insert!(neww, node, -w)
        end
    end
    return LinearStencil{d}(neww)
end # Base.:-()


# ------------------------------------------------------------------------------
function Base.:-(stc::LinearStencil{d}, a::T)::LinearStencil where {d,T <: Real}
    return LinearStencil{d}(stc.weights .- a)    
end # Base.:-()
function Base.:-(a::T, stc::LinearStencil{d})::LinearStencil where {d,T <: Real}
    return LinearStencil{d}(a .- stc.weights)    
end # Base.:-()


# ------------------------------------------------------------------------------
"""
    Base.:*(a::T, stc::LinearStencil{d})::LinearStencil{d}

Multiplication of a scalar `a` and a linear stencil `stc`. The weights of the
new stencil are the product of the scalar and the weights of the original sten-
cil.
"""
function Base.:*(a::T, stc::LinearStencil{d}) where {T,d}
    return LinearStencil{d}(a .* stc.weights)
end # Base.:*()
function Base.:*(stc::LinearStencil{d}, a::T) where {T,d}
    return LinearStencil{d}(a .* stc.weights)
end # Base.:*()


# ------------------------------------------------------------------------------
"""
    Base.:/(a::T, stc::LinearStencil{d})::LinearStencil{d}

Division of a scalar `a` and a linear stencil `stc`. The weights of the new st-
encil are the division of the scalar and the weights of the original stencil.
"""
function Base.:/(a::T, stc::LinearStencil{d}) where {T,d}
    return LinearStencil{d}(a ./ stc.weights)
end # Base.:/()
"""
    Base.:/(stc::LinearStencil{d}, a::T)::LinearStencil{d}

Division of a linear stencil `stc` and a scalar `a`. The weights of the new st-
encil are the division of the weights of the original stencil and the scalar.
"""
function Base.:/(stc::LinearStencil{d}, a::T) where {T,d}
    return LinearStencil{d}(stc.weights ./ a)
end # Base.:/()


# ------------------------------------------------------------------------------
"""
    apply(
        stc::LinearStencil{d}, 
        G::AdaptiveSparseGrid{d} ;
        applyto::Symbol = :nodal
    )::Float64

Apply the linear stencil `stc` to the grid `G`. The result is the sum of the
product of the weights and the function values of the nodes in the stencil.

The `applyto` argument can be either `:nodal` or `:hierarchical`. If `:nodal`,
then the function values of the nodes are used. If `:hierarchical`, then the
hierarchical coefficients of the nodes are used.
"""
function apply(
    stc::LinearStencil{d}, 
    G::AdaptiveSparseGrid{d} ;
    applyto::Symbol = :nodal
)::Float64 where d
    res = 0.0
    if applyto == :nodal
        for (node, w) in pairs(stc.weights)
            res += w * G.nv[node].f
        end
    elseif applyto == :hierarchical
        for (node, w) in pairs(stc.weights)
            res += w * G.nv[node].α
        end
    else
        throw(ArgumentError("invalid applyto: $applyto"))
    end
    return res
end # apply()


# ------------------------------------------------------------------------------
"""
    stencil2jv(
        stc::LinearStencil{d}, 
        node2index::Dictionary{Node{d},Int}
    )::Tuple{Vector{Int}, Vector{Float64}}

Convert a linear stencil to a pair of vectors `(Js, Vs)`. which can be used to 
define a sparse vector `SparseArrays.SparseVector{Float64,Int}`. The `Js` vecto-
r contains the indices of the nodes in the stencil, and the `Vs` vector contains
the weights of the nodes in the stencil.

## Notes
- This function requires a mapping from the nodes to their indices in ASG. It is
designed for the scenario of repeatingly applying stencils to the same ASG such
that the mapping can be reused.
"""
function stencil2jv(
    stc::LinearStencil{d}, 
    node2index::Dictionary{Node{d},Int}
)::Tuple{Vector{Int}, Vector{Float64}} where d
    Js = []; Vs = []
    for (node, w) in pairs(stc.weights)
        push!(Js, node2index[node])
        push!(Vs, w)
    end
    return (Js, Vs)
end # stencil2jv()

"""
    stencil2jv(
        stc::LinearStencil{d}, 
        G::AdaptiveSparseGrid{d}
    )::Tuple{Vector{Int}, Vector{Float64}}

A wrapper function for `stencil2jv(stc, node2index)`. The `node2index` is gener-
ated as `Dictionary{Node{d},Int}(keys(G.nv), 1:length(G.nv))`.
"""
function stencil2jv(
    stc::LinearStencil{d}, 
    G::AdaptiveSparseGrid{d}
)::Tuple{Vector{Int}, Vector{Float64}} where d
    node2index = Dictionary{Node{d},Int}(keys(G.nv), 1:length(G.nv))
    return stencil2jv(stc, node2index)
end # stencil2jv()