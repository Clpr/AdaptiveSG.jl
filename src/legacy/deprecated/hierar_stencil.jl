# Operations/arithmetics on HierarchicalBasisStencil{d}



# ------------------------------------------------------------------------------
"""
    Base.:(==)(hbs1::LinearStencil{d}, hbs2::LinearStencil{d})::Bool

Equality comparison of two hierarchical basis stencils. Two stencils are equal 
if every field in the two stencils are (value) equal.
"""
function Base.:(==)(
    hbs1::HierarchicalBasisStencil{d},
    hbs2::HierarchicalBasisStencil{d}
)::Bool where d
    if hbs1.cst != hbs2.cst
        return false
    end
    if hbs1.val != hbs2.val
        return false
    end
    if any(hbs1.stc .!= hbs2.stc)
        return false
    end
end


# ------------------------------------------------------------------------------
"""
    Base.:+(
        hbs1::HierarchicalBasisStencil{d}, 
        hbs2::HierarchicalBasisStencil{d}
    )::HierarchicalBasisStencil{d}

Addition of two hierarchical basis stencils. Stencil, constant, and value add up
respectively:

`val1 + val2 = transpose(stc1 + stc2) * H * F + (cst1 + cst2)`
"""
function Base.:+(
    hbs1::HierarchicalBasisStencil{d},
    hbs2::HierarchicalBasisStencil{d}
)::HierarchicalBasisStencil{d} where d
    # stencil, constant, and value add up respectively
    return HierarchicalBasisStencil{d}(
        hbs1.stc .+ hbs2.stc,
        hbs1.cst  + hbs2.cst,
        hbs1.val  + hbs2.val
    )
end 


# ------------------------------------------------------------------------------
"""
    Base.:+(
        hbs::HierarchicalBasisStencil{d}, 
        a::T
    )::HierarchicalBasisStencil{d} where {d,T <: Real}

Addition of a scalar `a` and a hierarchical basis stencil `hbs`. The constant
field of the new stencil is the sum of the scalar and the `val`, `cst` field of 
the original stencil, while the stencil field remain the same.
"""
function Base.:+(
    hbs::HierarchicalBasisStencil{d},
    a::T
)::HierarchicalBasisStencil{d} where {d,T <: Real}
    return HierarchicalBasisStencil{d}(
        hbs.stc,
        hbs.cst + a,
        hbs.val + a
    )
end # Base.:+
function Base.:+(
    a::T,
    hbs::HierarchicalBasisStencil{d}
)::HierarchicalBasisStencil{d} where {d,T <: Real}
    return HierarchicalBasisStencil{d}(
        hbs.stc,
        hbs.cst + a,
        hbs.val + a
    )
end # Base.:+


# ------------------------------------------------------------------------------
function Base.:-(
    hbs1::HierarchicalBasisStencil{d}, 
    hbs2::HierarchicalBasisStencil{d}
)::HierarchicalBasisStencil{d} where d
    return HierarchicalBasisStencil{d}(
        hbs1.stc .- hbs2.stc,
        hbs1.cst  - hbs2.cst,
        hbs1.val  - hbs2.val
    )
end # Base.:-
function Base.:-(
    hbs::HierarchicalBasisStencil{d}, 
    a::T
)::HierarchicalBasisStencil{d} where {d,T <: Real}
    return HierarchicalBasisStencil{d}(
        hbs.stc,
        hbs.cst - a,
        hbs.val - a
    )
end # Base.:-
function Base.:-(
    a::T,
    hbs::HierarchicalBasisStencil{d}
)::HierarchicalBasisStencil{d} where {d,T <: Real}
    return HierarchicalBasisStencil{d}(
        -hbs.stc,
        a - hbs.cst,
        a - hbs.val
    )
end # Base.:-


# ------------------------------------------------------------------------------
function Base.:*(
    a::T,
    hbs::HierarchicalBasisStencil{d}
)::HierarchicalBasisStencil{d} where {d,T <: Real}
    return HierarchicalBasisStencil{d}(
        a .* hbs.stc,
        a * hbs.cst,
        a * hbs.val
    )
end # Base.:*
function Base.:*(
    hbs::HierarchicalBasisStencil{d},
    a::T
)::HierarchicalBasisStencil{d} where {d,T <: Real}
    return HierarchicalBasisStencil{d}(
        a .* hbs.stc,
        a * hbs.cst,
        a * hbs.val
    )
end # Base.:*


# ------------------------------------------------------------------------------
function Base.:/(
    hbs::HierarchicalBasisStencil{d},
    a::T
)::HierarchicalBasisStencil{d} where {d,T <: Real}
    return HierarchicalBasisStencil{d}(
        hbs.stc ./ a,
        hbs.cst / a,
        hbs.val / a
    )
end # Base.:/


