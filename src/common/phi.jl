export ϕ


# ------------------------------------------------------------------------------
"""
    ϕ(x::Float64)::Float64

Return the one-dimensional hat function value at `x`, `x` non-trivial in [-1,1].

`ϕ := 1 - |x|` if `x` is in [-1,1], and `ϕ := 0` otherwise.

## Notes
- This function doesn't specially handle the case when `l=2` and `i=0` or `i=2` 
(boundary nodes). Use this function only if you know what you are doing. Otherw-
ise, use `ϕ(x::Float64, l::Int, i::Int)` instead.
- If `x` is `NaN`, then `ϕ(x)` returns 0.0
"""
function ϕ(x::Float64)::Float64
    return (-1.0 <= x <= 1.0) ? 1.0 - abs(x) : 0.0
end


# ------------------------------------------------------------------------------
"""
    ϕ(x::Float64, l::Int, i::Int)::Float64

Return the one-dimensional hat function value at `x` given base point `(l,i)`.

## Notes
- This function specially handles the case when `l=2` and `i=0` or `i=2` (bound-
ary nodes). Use this function primarily.
"""
function ϕ(x::Float64, l::Int, i::Int)::Float64
    if l > 2
        return ϕ(x * power2(l - 1) - i)
    elseif (l == 2) && (i == 0)
        return (0 <= x <= 0.5) ? (1.0 - 2.0 * x) : 0.0
    elseif (l == 2) && (i == 2)
        return (0.5 <= x <= 1) ? (2.0 * x - 1.0) : 0.0
    elseif l == 1
        return (0 <= x <= 1) ? 1.0 : 0.0
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
end


# ------------------------------------------------------------------------------
"""
    ϕ(x::T, node::Node{d})::Float64 where {d,T<:AbstractVector{Float64}}

Compute the tensor product basis function value at a `d`-dim point `x` which is
denoted as a vector of `d` elements; given the `d`-dim node as the base point.

## Notes
- This function is safe. So length check is performed.
- If you need a more radical version that shuts down all the safety checks, use
`ϕ_unsafe(x::T, node::Node{d})::Float64` instead. This function is not exported.
"""
function ϕ(x::T, node::Node{d})::Float64 where {d,T<:AbstractVector{Float64}}
    if length(x) != d
        throw(ArgumentError("length of x is not equal to d"))
    end
    val = 1.0; for i in 1:d; val *= ϕ(x[i], node.ls[i], node.is[i]); end
    return val
end


# ------------------------------------------------------------------------------
"""
    ϕ_unsafe(x::T, node::Node{d})::Float64 where {d,T<:AbstractVector{Float64}}

Unsafe version of `ϕ(x::T, node::Node{d})::Float64`. It uses `@inbounds` macro.
Use this function only if you know what you are doing.

## Notes
- This function is not exported.
- This unsafe version has a feature that it ignores all elements of `x` that are
beyond position `d`. So it is safe to pass a vector of length `d+1` or more.
"""
function ϕ_unsafe(
    x::T, 
    node::Node{d}
)::Float64 where {d,T<:AbstractVector{Float64}}
    val = 1.0
    @inbounds for i in 1:d
        val *= ϕ(x[i], node.ls[i], node.is[i])
    end
    return val
end
