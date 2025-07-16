#===============================================================================
MATH HELPERS

- generic math helper functions
- basis function ϕ and its variants
- node arithmetics
- node tree/hierachy growing helpers
===============================================================================#
export nodedepth
export perturb
export scale

export ϕ

export isvalid
export get_x
export isboundary, boundaryflag
export allnodes1d
export allindices_of_level

export child, parent
export ghost_neighbor

#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SECTION: Generic math helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
"""
    maybe_threads(flag::Bool, expr::Expr)

Wrap an expression `expr` with `Threads.@threads` macro if `flag` is `true`.

Usage: `@maybe_threads true for i in .....`
"""
macro maybe_threads(flag, expr)
    quote
        if $(flag)
            Threads.@threads $expr
        else
            $expr
        end
    end |> esc
end # maybe_threads
# ------------------------------------------------------------------------------
"""
    power2(x::Int)::Int

Compute the integer power of 2, `2^x`, raised to the given integer `x`.

# Arguments
- `x::Int`: The exponent of the power of 2.

# Returns
- `Int`: The result of 2 raised to the power of `x`.

# Notes
- For performance, use the bitwise left shift operator
- Useful in node arithmetics
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
end
function nodedepth(ls::AbstractVector{Int})::Int
    return sum(ls) - length(ls) + 1
end
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
# ------------------------------------------------------------------------------
"""
    unitvec(d::Int, j::Int)::Vector{Float64}

Return a unit vector of dimension `d` with the `j`-th element being 1.0 and all
"""
function unitvec(d::Int, j::Int)::Vector{Float64}
    v = zeros(Float64, d); v[j] = 1.0; return v
end # unitvec()
# ------------------------------------------------------------------------------
"""
    spunitvec(d::Int, j::Int)::SparseVector{Float64, Int}

Return a sparse unit vector of dimension `d` with the `j`-th element being 1.0.
"""
function spunitvec(d::Int, j::Int)::SparseVector{Float64, Int}
    return sparsevec(Int[j,], Float64[1.0,], d)
end # spunitvec()
# ------------------------------------------------------------------------------
"""
    perturb(x::AbstractVector, j::Int, vnew::Real)::AbstractVector

Return a new vector `x2` by perturbing the `j`-th element of the input vector 
`x` with the new value `vnew`. The function does not modify the input vector `x`
but require the type of `x` to be mutable.
"""
function perturb(x::AbstractVector, j::Int, vnew::Real)::AbstractVector
    x2 = copy(x)
    x2[j] = vnew
    return x2
end # perturb()
# ------------------------------------------------------------------------------
"""
    isbig2add(
        α       ::Float64, 
        fx      ::Float64, 
        rtol    ::Float64,
        atol    ::Float64,
        use_rtol::Bool
    )::Bool

Check if the given coefficient `α` is big enough to add a new node, using relat-
ive tolerance `rtol`.

## Notes
- the check is scaled by the function value at the candidate child node
- multi-linear interpolation implies phi = 1 at the grid point, so the nodal coe
fficient and the hierarchical coefficient is comparable in magnitude, which can
be conceptually seen as 1 - R^2
- we specically deal with the case `fx ≈ 0` by using absolute tolerance instead.
The critera of applying the atol is `isapprox(fx, 0.0, atol = 1E-8)`, in which
`1E-8` is chosen because it is the machine epsilon of `Float32` which is common
in GPU computing. One implication is that if your function curve/surface/field
crosses the zero line many times, then it is expected to have more nodes than
usual. This feature might be useful for some applications (e.g. caring about the
system's behavior near the equilibrium point) but it may also be a drawback for
some other applications (e.g. a wave surface that crosses the zero line many ti-
mes). If you expect it is a drawback for your application, then there are two
solutions: 1) plus a constant to the function to push it away from zero; 2) Let 
the training process accept absolute tolerance (planed in the future).
"""
function isbig2add(
    α       ::Float64, 
    fx      ::Float64, 
    rtol    ::Float64,
    atol    ::Float64,
    use_rtol::Bool
)::Bool
    if use_rtol
        if isapprox(fx, 0.0, atol = rtol)
            return abs(α) > rtol
        else
            return (abs(α) / abs(fx)) > rtol
        end
    else
        # NOTES: the `atol` is defined in the mean of the residual fit coef
        return abs(α) > atol
    end
end # isbig2add()
# ------------------------------------------------------------------------------
"""
    scale(
        x::AbstractVector, 
        from_lb, 
        from_ub ; 
        to_lb = 0.0,
        to_ub = 1.0,
    )

Scales a N-dimensional vector/point `x` from the range `[from_lb, from_ub]`
to the range `[to_lb, to_ub]`. The scaling is done element-wise.

Returns a `Vector{Float64}` of the same length as `x`.

## Notes
- The `from_lb` and `from_ub` can be either a scalar or a vector of the same
length as `x`. If they are scalars, then the same value is used for all elements
of `x`.
- The `to_lb` and `to_ub` can also be either a scalar or a vector of the same
length as `x`. If they are scalars, then the same value is used for all elements
of `x`.
- The function does not check if `from_lb` is less than `from_ub` or if `to_lb`
is less than `to_ub`. It is the user's responsibility to ensure that the input
values are valid.
"""
function scale(
    x::AbstractVector, 
    from_lb, 
    from_ub ; 
    to_lb = 0.0,
    to_ub = 1.0,
)::Vector{Float64}
    return to_lb .+ (to_ub .- to_lb) .* (x .- from_lb) ./ (from_ub .- from_lb)
end # scale






#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SECTION: Basis function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
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
    ϕ(
        x      ::AbstractVector ,
        levels ::AbstractVector ,
        indices::AbstractVector ;

        safe::Bool = false,
    )::Float64

Multi-dimenisonal tensor basis function value at a `d`-dim point `x` which is
defined in the hypercube [0,1]^d; given the `d`-dim reference point as the base
point, which is defined by the `levels` and `indices` vectors.

This is the main API to call in practice.

The `safe` argument is used to enable/disable the dimenision check.
"""
function ϕ(
    x      ::AbstractVector ,
    levels ::AbstractVector ,
    indices::AbstractVector ;

    safe::Bool = false,
)::Float64
    D = length(x)
    if safe
        @assert length(levels) == D "levels must match the dimension of x."
        @assert length(indices) == D "indices must match the dimension of x."
    end

    # apply ϕ to each dimension then tensor joined
    return ϕ.(x, levels, indices) |> prod
end








#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SECTION: Node arithmetics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
"""
    isvalid(Ls::Vector{Int}, Is::Vector{Int})::Bool

Check if the given `d`-dim node is valid.

This function is used to exclude the out-of-boundary nodes grown by (l=2,i=0) to
wards the left and (l=2,i=2) towards the right.

## Rules
- for all dimensions, level > 0
    - for a dimension, if level == 1, then index == 1 only
    - for a dimension, if level == 2, then index == 0 or 2 only
    - for a dimension, if level > 2, then index is odd and positive
- `depth > 0` must be true
"""
function isvalid(Ls::Vector{Int}, Is::Vector{Int})::Bool
    # check dimension
    d = length(Ls)
    if d < 1; return false; end
    if length(Is) != d; return false; end

    # check: levels & indices by dimension
    for j in 1:d
        @inbounds if Ls[j] < 1
            return false
        elseif (Ls[j] == 1)
            if Is[j] != 1
                return false
            end
        elseif Ls[j] == 2
            if (Is[j] != 0) & (Is[j] != 2)
                return false
            end
        else
            if (Is[j] < 1) | (!isodd(Is[j]))
                return false
            end
        end # if
    end # j

    # check: depth
    depth = nodedepth(Ls)
    if depth < 1; return false; end

    return true
end # isvalid()
# ------------------------------------------------------------------------------
"""
    get_x(l::Int, i::Int)::Float64

Return the one-dimensional hierarchical grid node value for a given level-index
pair `(l, i)`. The grid node is in a unit hypercube [0,1].

## Notes
- no scaling
- given `l`, `i` = 1,2,...,2^(l-1)-1
- private function
- no boundary checks
- no error handling
- error thrown for invalid level-index pair
"""
function get_x(l::Int, i::Int)::Float64
    if (l > 2) & isodd(i)
        return Float64(i / power2(l - 1))
    elseif (l == 2) & (i == 0)
        return 0.0
    elseif (l == 2) & (i == 2)
        return 1.0
    elseif (l == 1) & (i == 1)
        return 0.5
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
end
# ------------------------------------------------------------------------------
"""
    get_x(
        Ls::AbstractVector{Int}, 
        Is::AbstractVector{Int} ;
        lb = 0.0,
        ub = 1.0,
    )

Return the hierarchical grid point value for a given `d`-dim node, scaled to the
range `[lb, ub]`. The point is in a unit hypercube [0,1]^d, where `d` is the
length of `Ls` and `Is`.
"""
function get_x(
    Ls::AbstractVector{Int}, 
    Is::AbstractVector{Int} ;
    lb = 0.0,
    ub = 1.0,
)
    return scale(get_x.(Ls, Is), 0.0, 1.0, to_lb = lb, to_ub = ub)
end
# ------------------------------------------------------------------------------
"""
    fraction2li(i2j::Rational)::NTuple{2,Int}

Convert the given 1-dim node realization denoted as a fraction `i2j` to its lev-
el-index pair `(l,i)`.

## Notes
- Julia's `Rational` does automatically fraction reduction
- This function is useful in finding the neighbor nodes of a specific depth
- This function is useful in listing all the nodes until a specific depth
- The nominator is the i-th node in the ascending order of the cummulative nodes
at all levels. the denominator is the total number of nodes cummulated until the
level. e.g. `1//2` is the root point, `2//2` is the right boundary, `0//2` is
the left boundary. The denominator is always a power of 2, i.e. `2^(l-1)`.
- e.g. `fraction2li.( (0:4) .// 4 )` returns `((2,0),(3,1),(1,1),(3,3),(2,2))`,
which is the cummulative node list until depth 3 (4 = 2^(3-1)).
"""
function fraction2li(i2j::Rational)::NTuple{2,Int}
    if i2j == 0//1
        # case: left boundary
        return (2, 0)
    elseif i2j == 1//1
        # case: right boundary
        return (2, 2)
    elseif i2j == 1//2
        # case: root point
        return (1, 1)
    else
        # case: interior point of level > 2
        return (invpower2(i2j.den) + 1, i2j.num)
    end
end
# ------------------------------------------------------------------------------
"""
    perturb(
        Ls  ::AbstractVector{Int}, 
        Is  ::AbstractVector{Int}, 
        dims::Int, 
        l   ::Int, 
        i   ::Int
    )::NTuple{2,Vector{Int}}

Return a new `d`-dim node by perturbing the given node along the `dims` dimensi-
on with the new level `l` and index `i`.
"""
function perturb(
    Ls  ::AbstractVector{Int}, 
    Is  ::AbstractVector{Int}, 
    dims::Int, 
    l   ::Int, 
    i   ::Int,
)::NTuple{2,Vector{Int}}
    newL = Ls |> copy |> Vector{Int}
    newI = Is |> copy |> Vector{Int}
    newL[dims] = l
    newI[dims] = i
    return (newL, newI)
end
# ------------------------------------------------------------------------------
"""
    isboundary(l::Int, i::Int)::Bool

Check if the given 1-dimensional node of level-index pair `(l, i)` is a boundary
node (i.e. left or right boundary). Returns `true` if it is a boundary node,
`false` otherwise.

## Notes
- To check if a `d`-dim node is a boundary node, use `isboundary.(Ls, Is)` that
returns a `BitVector` of the same length as `Ls` and `Is`.
- This method does not validate the level-index pair `(l, i)`. It is the user's
responsibility to ensure that the input values are valid.
- This method basically checks if the node is one of (2,0) or (2,2).
"""
function isboundary(l::Int, i::Int)::Bool
    if l == 2
        return (i == 0) || (i == 2)
    else
        return false
    end
end
# ------------------------------------------------------------------------------
"""
    boundaryflag(l::Int, i::Int)::Int

Return the boundary flag of the given 1-dimensional node of level-index pair
`(l, i)`. The flag is defined as follows:
- -1: left boundary (i.e. (2,0))
-  1: right boundary (i.e. (2,2))
-  0: not a boundary node but interior

## Notes
- To check if a `d`-dim node is a boundary node, use `boundaryflag.(Ls, Is)`
that returns a `Vector{Int}` of the same length as `Ls` and `Is`.
- This method does not validate the level-index pair `(l, i)`. It is the user's
responsibility to ensure that the input values are valid.
"""
function boundaryflag(l::Int, i::Int)::Int
    if l == 2
        if i == 0
            return -1 # left boundary
        elseif i == 2
            return 1 # right boundary
        else
            throw(ArgumentError("invalid level-index pair ($l, $i)"))
        end
    else
        return 0 # not a boundary node but interior
    end
end
# ------------------------------------------------------------------------------
"""
    allnodes1d(l::Int)::NTuple{2,Vector{Int}}

Return all the 1-dim nodal nodes in the FULL hierarchical grid of level `l` in
the ascending order of the node value.

## Notes
- This function is used for generating the full nodal grid for testing or other
purposes.
- To get the realized node values in [0,1], use `get_x.(allnodes1d(l)...)`.
- To get the indices of a specific level, use `allindices_of_level(l)`.
"""
function allnodes1d(l::Int)::NTuple{2,Vector{Int}}
    if l < 1
        throw(ArgumentError("level must be >= 1 but got $l"))
    elseif l == 1
        return ([1,],[1,])
    elseif l == 2
        return ([1,2,2],[1,0,2])
    else
        Ls = Int[]
        Is = Int[]
        den = power2(l - 1)
        for i in 0:power2(l-1)
            lnew, inew = fraction2li(i // den)
            push!(Ls, lnew)
            push!(Is, inew)
        end
        return (Ls, Is)
    end
end
# ------------------------------------------------------------------------------
"""
    allindices_of_level(l::Int)::Vector{Int}

Return all the indices of the regular sparse nodes of level `l`.

## Notes
- If `l == 1`, then only the root node (index 1) is returned.
- If `l == 2`, then only the two boundary nodes (index 0 and 2) are returned.
- If `l > 2`, then all the odd indices from 1 to 2^(l-1)-1 are returned.
"""
function allindices_of_level(l::Int)::Vector{Int}
    if l > 2
        return collect(1:2:(power2(l-1) - 1))
    elseif l == 2
        return Int[0, 2]
    elseif l == 1
        return Int[1,]
    else
        throw(ArgumentError("invalid level $l"))
    end
end


















#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SECTION: Node tree/hierachy growing helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
"""
    _ghost_left(
        Ls  ::AbstractVector{Int},
        Is  ::AbstractVector{Int}, 
        dims::Int, 
        l0  ::Int
    )::NTuple{2,Vector{Int}}

Return the left nodal point of the given `d`-dim node along `dims` dimension in 
a `l0`-level full hierarchical grid (nodal grid).

## Notes
- This function can be used to find the "ghost" node in the finite difference ap
  plications.
- The left nodal point is the closest left nodal point in the full hierarchical 
  grid of level `l0`.
- Check the source code for the detailed explanation of the algorithm. 
- If there is no lefter nodal point defined, this function returned an invalid
node with `ret.ls[dims]==0` and `ret.is[dims]==-1`.
- We assume the input (Ls,Is) represents a valid mutli-dimensional node
"""
function _ghost_left(
    Ls  ::AbstractVector{Int},
    Is  ::AbstractVector{Int}, 
    dims::Int, 
    l0  ::Int
)::NTuple{2,Vector{Int}}
    # the one-dim node splited from the `dims`-dim of `node`
    lj = Ls[dims]; ij = Is[dims]

    # discussion by case: l0 --> lj --> ij, nested
    if l0 < 1
        throw(ArgumentError("invalid level $l0"))
    elseif l0 == 1
        # only the root node exist, no left point in any case
        lnew = 0; inew = -1
    elseif l0 == 2
        if (lj == 2) & (ij == 0)
            # the left boundary point, no lefter point
            lnew = 0; inew = -1
        elseif (lj == 2) & (ij == 2)
            # the right boundary point, the left point is the root
            lnew = 1; inew = 1
        elseif (lj == 1) & (ij == 1)
            # the root (center) point, its left point is the left boundary
            lnew = 2; inew = 0
        else
            # invalid points
            lnew = 0; inew = -1
        end
    elseif l0 > 2
        if lj < 1
            throw(ArgumentError("lj<1, invalid input node"))
        elseif lj == 1
            if ij == 1
                # the root point, whose definition is different from lj>2
                # denote the lefter point as a fraction of level `l0`
                _lp = (power2(l0 - 2) - 1) // power2(l0 - 1)
                # then, convert it to the level-index pair
                lnew, inew = fraction2li(_lp)
            else
                throw(ArgumentError("invalid input node"))
            end
        elseif lj == 2
            if ij == 0
                # the left boundary point, no lefter point
                lnew = 0; inew = -1
            elseif ij == 2
                # the right boundary point, the lefter point is the largest inte
                # ior point of level l0
                lnew = l0; inew = power2(l0 - 1) - 1
            else
                throw(ArgumentError("invalid input node"))
            end
        elseif 2 < lj <= l0
            _lp = (ij * power2(l0 - lj) - 1) // power2(l0 - 1)
            lnew, inew = fraction2li(_lp)
        else
            throw(ArgumentError("lj > l0 found, no grid defined"))
        end
    end # if l0

    return perturb(Ls, Is, dims, lnew, inew)
end
# ------------------------------------------------------------------------------
"""
    _ghost_right(
        Ls  ::AbstractVector{Int},
        Is  ::AbstractVector{Int}, 
        dims::Int, 
        l0  ::Int
    )::NTuple{2,Vector{Int}}

Return the right nodal point of the given `d`-dim node along `dims` dimension in
a `l0`-level full hierarchical grid (nodal grid).

## Notes
- This function can be used to find the "ghost" node in the finite difference ap
  plications.
- The right nodal point is the nearest right nodal point in the full hierarchicl
grid of level `l0`.
"""
function _ghost_right(
    Ls  ::AbstractVector{Int},
    Is  ::AbstractVector{Int}, 
    dims::Int, 
    l0  ::Int
)::NTuple{2,Vector{Int}}
    # the one-dim node splited from the `dims`-dim of `node`
    lj = Ls[dims]; ij = Is[dims]

    # discussion by case: l0 --> lj --> ij, nested
    if l0 < 1
        throw(ArgumentError("invalid level $l0"))
    elseif l0 == 1
        # only the root node exist, no righter point in any case
        lnew = 0; inew = -1
    elseif l0 == 2
        if (lj == 2) & (ij == 0)
            # the left boundary point, its righter point is the root
            lnew = 1; inew = 1
        elseif (lj == 2) & (ij == 2)
            # the right boundary point, no righter point
            lnew = 0; inew = -1
        elseif (lj == 1) & (ij == 1)
            # the root (center) point, its righter point is the right boundary
            lnew = 2; inew = 2
        else
            # invalid points
            lnew = 0; inew = -1
        end
    elseif l0 > 2
        if lj < 1
            throw(ArgumentError("lj<1, invalid input node"))
        elseif lj == 1
            if ij == 1
                # the root point, whose definition is different from lj>2
                # denote the righter point as a fraction of level `l0`
                _lp = (power2(l0 - 2) + 1) // power2(l0 - 1)
                # then, convert it to the level-index pair
                lnew, inew = fraction2li(_lp)
            else
                throw(ArgumentError("invalid input node"))
            end
        elseif lj == 2
            if ij == 0
                # the left boundary point, the righter point is the 1st interior
                # point of level l0
                lnew = l0; inew = 1
            elseif ij == 2
                # the right boundary point, no righter point
                lnew = 0; inew = - 1
            else
                throw(ArgumentError("invalid input node"))
            end
        elseif 2 < lj <= l0
            _lp = (ij * power2(l0 - lj) + 1) // power2(l0 - 1)
            lnew, inew = fraction2li(_lp)
        else
            throw(ArgumentError("lj > l0 found, no grid defined"))
        end
    end # if l0

    return perturb(Ls, Is, dims, lnew, inew)
end
# ------------------------------------------------------------------------------
"""
    ghost_neighbor(
        Ls  ::AbstractVector{Int},
        Is  ::AbstractVector{Int},
        dims::Int,
        l0  ::Int,
        side::Symbol = :left,
    )::NTuple{2,Vector{Int}}

Return the ghost neighbor node of the given `d`-dim node along `dims` dimension
in a `l0`-level full hierarchical grid (nodal grid). The `side` argument does
specify which ghost neighbor to return, either `:left` or `:right`.


"""
function ghost_neighbor(
    Ls  ::AbstractVector{Int},
    Is  ::AbstractVector{Int},
    dims::Int,
    l0  ::Int,
    side::Symbol = :left,
)::NTuple{2,Vector{Int}}
    if side == :left
        return _ghost_left(Ls, Is, dims, l0)
    elseif side == :right
        return _ghost_right(Ls, Is, dims, l0)
    else
        throw(ArgumentError("side must be :left or :right"))
    end
end
# ------------------------------------------------------------------------------
"""
    ghost_distance(lmax::Int)::Float64

Return the ghost mesh step size `h` of the underlying nodal grid, given the max-
imum level `lmax`. The ghost mesh step size is the mesh step size of the largest
level. In a multi-dimensional case the largest level is NOT equal to the depth 
of the ASG tree but along each dimension.
"""
function ghost_distance(lmax::Int)::Float64
    return 1.0 / power2(lmax - 1)
end
# ------------------------------------------------------------------------------
"""
    _child_left(
        Ls  ::AbstractVector{Int},
        Is  ::AbstractVector{Int},
        dims::Int
    )::NTuple{2,Vector{Int}}

Return the left child node of the given `d`-dim node along `dims` dimension.

## Notes
* when (l == 2) && (i == 0), the `i` of the left child is -1 which is invalid
* when (l == 2) && (i == 2), the `i` of the left child is 3 which is invalid
* does not throw an error if (l == 2) and (i != 0 or 2). But the returned node
is invalid and can be checked by `isvalid()`.
"""
function _child_left(
    Ls  ::AbstractVector{Int},
    Is  ::AbstractVector{Int},
    dims::Int
)::NTuple{2,Vector{Int}}
    l = Ls[dims]
    i = Is[dims]
    if l > 2
        inew = 2 * i - 1
    elseif (l == 2) && (i == 0)
        inew = -1 # invalid index, will be used to exclude the node
    elseif (l == 2) && (i == 2)
        inew = 3
    elseif l == 1
        inew = 0
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
    return perturb(Ls, Is, dims, l + 1, inew)
end
# ------------------------------------------------------------------------------
"""
    _child_right(
        Ls  ::AbstractVector{Int},
        Is  ::AbstractVector{Int},
        dims::Int
    )::NTuple{2,Vector{Int}}

Return the right child node of the given `d`-dim node along `dims` dimension.

## Notes
* check the docstring of `get_child_left` for the invalid cases
"""
function _child_right(
    Ls  ::AbstractVector{Int},
    Is  ::AbstractVector{Int},
    dims::Int
)::NTuple{2,Vector{Int}}
    l = Ls[dims]
    i = Is[dims]
    if l > 2
        inew = 2 * i + 1
    elseif (l == 2) && (i == 0)
        inew = 1
    elseif (l == 2) && (i == 2)
        inew = -1  # invalid index, will be used to exclude the node
    elseif l == 1
        inew = 2
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
    return perturb(Ls, Is, dims, l + 1, inew)
end
# ------------------------------------------------------------------------------
"""
    child(
        Ls  ::AbstractVector{Int},
        Is  ::AbstractVector{Int},
        dims::Int,
        side::Symbol = :left,
    )::NTuple{2,Vector{Int}}

Return the child node of the given `d`-dim node along `dims` dimension.
The `side` argument specifies which child to return, either `:left` or `:right`.
"""
function child(
    Ls  ::AbstractVector{Int},
    Is  ::AbstractVector{Int},
    dims::Int ;
    side::Symbol = :left,
)::NTuple{2,Vector{Int}}
    if side == :left
        return _child_left(Ls, Is, dims)
    elseif side == :right
        return _child_right(Ls, Is, dims)
    else
        throw(ArgumentError("side must be :left or :right"))
    end
end
# ------------------------------------------------------------------------------
"""
    parent(
        Ls  ::AbstractVector{Int},
        Is  ::AbstractVector{Int},
        dims::Int
    )::NTuple{2,Vector{Int}}

Return the parent node of the given `d`-dim node along `dims` dimension.

## Notes
* special case: l == 3: (3,1) -> (2,0), (3,3) -> (2,2)
* special case: l == 2: (2,0) -> (1,1), (2,2) -> (1,1)
* special case: l == 1: (1,0) -> (0,0)
"""
function parent(
    Ls  ::AbstractVector{Int},
    Is  ::AbstractVector{Int},
    dims::Int
)::NTuple{2,Vector{Int}}
    l = Ls[dims]
    i = Is[dims]
    if l > 3
        inew = div(i, 4) * 2 + 1
    elseif (l == 3) && (i == 1)
        inew = 0
    elseif (l == 3) && (i == 3)
        inew = 2
    elseif (l == 2) && (i == 0)
        inew = 1
    elseif (l == 2) && (i == 2)
        inew = 1
    elseif l == 1
        inew = 0
    else
        throw(ArgumentError("invalid level-index pair ($l, $i)"))
    end
    return perturb(Ls, Is, dims, l - 1, inew)
end


