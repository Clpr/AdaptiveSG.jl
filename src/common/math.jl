export power2, invpower2, odduntil, nodedepth

# ------------------------------------------------------------------------------
"""
    dir(x::Any)

Return the field names of the given object `x`.

DELETE THIS FUNCTION AFTER DEBUGGING.
"""
dir(x::Any) = fieldnames(typeof(x))


# ------------------------------------------------------------------------------
"""
    power2(x::Int)::Int

Compute the power of 2, `2^x`, raised to the given integer `x`.

# Arguments
- `x::Int`: The exponent of the power of 2.

# Returns
- `Int`: The result of 2 raised to the power of `x`.
"""
function power2(x::Int)::Int
    return 2 << (x - 1)
end # power2()


# ------------------------------------------------------------------------------
"""
    invpower2(x::Int)::Int

Compute the inverse of the power of 2, `2^x`, raised to the given integer `x`.
If `x` is not a power of 2, the function throws an `InexactError`.
"""
function invpower2(x::Int)::Int
    return Int(log2(x))
end # power2inv()


# ------------------------------------------------------------------------------
"""
    odduntil(n::Int)::StepRange{Int, Int}

Return a collection of all odd integers up to `n` >= 1, starting from 1.
No boundary checks. if `n` < 1, the function returns an empty `Vector{Int}`
"""
odduntil(n::Int)::StepRange{Int, Int} = 1:2:n


# ------------------------------------------------------------------------------
"""
    nodedepth(ls::NTuple{d, Int})::Int where d

Return the depth (deepest level) of the given `d`-dim node, which is defined as 
`norm1(node.ls) - d + 1` where `norm1` is the L1 norm of the vector, i.e. sum of
the absolute value of the elements.

With this definition, for a `d`-dim node, the unique starting root point of the 
tree `(1,1,1,...)` always has depth 1.

Meanwhile, this definition allows the tree to grow in a dim-by-dim way, i.e. for
any node `(ls,is)`, its growth only considers children along each dimension. The
n, any `d`-dim node will have `2d` children (left/right * d). No need to create
the children by taking children along >1 dimensions at once (otherwise, one has
to consider `C^1_d + C^2_d + ... + C^d_d = 2^d` children).

A useful formula: for a `d`-dim grid, the number of nodes (if no adaption) at de
pth `l` is `(2^(l-1))^d` where l = 1,2,...,`max_depth`.

## Notes
- For special case `d=1`, depth = level
"""
function nodedepth(ls::NTuple{d, Int})::Int where d
    return sum(ls) - d + 1
end # depth()


# ------------------------------------------------------------------------------
"""
    spinv_lower(L::SparseMatrixCSC{Float64, Int})::SparseMatrixCSC{Float64, Int}

Compute the inverse of a lower triangular matrix `L` in a sparse format. The
inverse matrix is also in a sparse format.

## Notes
- This function assumes that `L` is a properly-defined lower triangular matrix.
There is no check for this condition.
- This function is needed because Julia does not provide a built-in function to
compute the inverse of a sparse matrix directly. The function is based on the
LU decomposition.
- IMPORTANT: Julia's sparse `lu` does NOT return the same result as the dense
`lu` function. It behvaes like the `[L,U,P,Q] = lu(A)` in MATLAB. The `L` & `U`
are not the same as the `L` and `U` in the dense `lu` function. The `L` and `U`
from the sparse `lu` are the lower and upper triangular matrices of the LU
decomposition, respectively. The `P` and `Q` are the row and column permutation
matrices, respectively.
"""
function spinv_lower(
    L::SparseMatrixCSC{Float64, Int}
)::SparseMatrixCSC{Float64, Int}
    n = size(L, 1)

    # Initialize the inverse matrix with zeros
    invL = spzeros(eltype(L), n, n)

    # Compute the diagonal elements
    for i in 1:n
        invL[i, i] = 1 / L[i, i]
    end

    # Compute the off-diagonal elements
    for i in 2:n
        for j in 1:i-1
            _total = 0.0
            for k in j:i-1
                if (L[i, k] != 0) && (invL[k, j] != 0)
                    _total += L[i, k] * invL[k, j]
                end
            end
            if _total != 0
                invL[i, j] = -_total / L[i, i]
            end
        end
    end

    return invL
end # spinv_lower()


# ------------------------------------------------------------------------------
"""
    spinv_upper(U::SparseMatrixCSC{Float64, Int})::SparseMatrixCSC{Float64, Int}

Compute the inverse of an upper triangular matrix `U` in a sparse format. The
inverse matrix is also in a sparse format.
"""
function spinv_upper(
    U::SparseMatrixCSC{Float64, Int}
)::SparseMatrixCSC{Float64, Int}
    n = size(U, 1)

    # Initialize the inverse matrix with zeros
    invU = spzeros(eltype(U), n, n)

    # Compute the diagonal elements
    for i in 1:n
        invU[i, i] = 1 / U[i, i]
    end

    # Compute the off-diagonal elements
    for i in n-1:-1:1
        for j in i+1:n
            _total = 0.0
            for k in i+1:j
                if (U[i, k] != 0) && (invU[k, j] != 0)
                    _total += U[i, k] * invU[k, j]
                end
            end
            if _total != 0
                invU[i, j] = -_total / U[i, i]
            end
        end
    end

    return invU
end # spinv_upper()









