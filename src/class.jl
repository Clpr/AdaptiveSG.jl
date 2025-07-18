#===============================================================================
DATA STRUCTURES

- abstract types
- generic sparse grid interpolant
    - type definition
    - overloaded methods
    - other interface methods
    - evaluation API
    - other methods
===============================================================================#
export AbstractSparseGridInterpolant
export SparseGridInterpolant
export maxlevels
export coefficients
export basis
export dehierarchization_matrix
export hierarchization_matrix



#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SECTION: Abstract types & type alias
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
abstract type AbstractSparseGridInterpolant{D} <: Any end

AbsVecTup = Union{Tuple, AbstractVector}
Point     = AbstractVector


#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SECTION: (Standard) sparse grid interpolant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
"""
    SparseGridInterpolant{D}

A standard sparse grid interpolant type for `D`-dimensional data. It contains
the domain information, nodes information, and coefficients information. The
nodes are defined in the unit hypercube `[0,1]^D` but the domain is used to
scale the nodes to the actual domain.

Many standard methods are overloaded for this type.

## Fields
- `lb::NTuple{D, Float64}`: lower bound of the domain
- `ub::NTuple{D, Float64}`: upper bound of the domain
- `levels::Matrix{Int}`: levels of the nodes, size `N * D`
- `indices::Matrix{Int}`: indices of the nodes, size `N * D`
- `depths::Vector{Int}`: depth of the nodes in the hierarchy/tree, size `N`
- `fvals::Vector{Float64}`: function values at the nodes, size `N`
- `coefs::Vector{Float64}`: interpolation/hierarchical coefficients, size `N`

## Example
```julia
asg = include("../src/AdaptiveSG.jl")

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

# Dimensionality: 3
D = 3

# Create an empty R^D --> R sparse grid interpolant in a custom domain
G = asg.SparseGridInterpolant(D, lb = 1:D, ub = 2:D+1)

# Create an empty R^D --> R sparse grid interpolant in the hypercube [0,1]^D
G = asg.SparseGridInterpolant(D)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

# Example function to fit
f2fit(X) = (X .* 2π) .|> sin |> sum

# usage 1: train the interpolant using isotropic regular sparse grid (RSG)
# accuracy level: 7; isotropic max level per dimension: 5
asg.train!(
    G, 
    f2fit, 
    7, 
    5,
    verbose = true, # display training progress
)


# usage 2: anisotropic regular sparse grid (RSG)
asg.train!(
    G, 
    f2fit, 
    7, 
    Tuple(6:D+6-1), # anisotropic max levels per dimension
    verbose = true, # display training progress
)


# usage 3: local refinement/adaption after the initial RSG training, using
#          adaptive sparse grid (ASG)
#          max depth allowed: 13, tol = 3E-3, tolerance type: absolute
#          parallel threads enabled
asg.adapt!(
    G, f2fit, 13,
    verbose  = true,
    tol      = 3e-3,
    toltype  = :absolute, # or toltype = :relative
    parallel = true,      # use parallel threads
)



# ------------------------------------------------------------------------------
# Useful operations
# ------------------------------------------------------------------------------

# Get: size information
length(G) |> println # number of nodes
ndims(G)  |> println # dimension of the grid, = 4
size(G)   |> println # (number of nodes, dimension)


# Check: if a point locates in the grid domain
rand(D) in G
rand(D) ∈ G
rand(D) ∉ G

# Clamp a point to the grid domain
clamp(rand(D) .* 10, G)

# Draw random points in the grid domain
rand(G)      # a random point in the grid domain
rand(G, 10)  # 10 random points in the grid domain

# Create a uniform grid along the 2nd dimension with 10 points
LinRange(G, 2, 10)

# Transform a point between the grid domain and the unit hypercube [0,1]^4
let x0 = rand(D)

    # hypercube --> grid domain
    x1 = asg.scale(x0, 0.0, 1.0, to_lb = G.lb, to_ub = G.ub)
    x2 = asg.scale(
        x0, 
        zeros(length(x0)), 
        ones(length(x0)), 
        to_lb = G.lb, 
        to_ub = G.ub
    )

    # grid domain --> hypercube
    x3 = asg.scale(x1, G.lb, G.ub)
    x4 = asg.scale(x2, G.lb, G.ub, to_lb = 0.0, to_ub = 1.0)

    [x0 x1 x2 x3 x4]
end

# Stack all the supporting grid nodes into a N*D matrix
nodes = stack(G)


# Get the maximum levels of the grid in the grid hierarchy/tree
asg.maxlevels(G)



# ------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------

# Serialize the interpolant to a compatability-maximized dictionary
dat = asg.serialize(G)

# De-serialize the dictionary to an interpolant
G2 = asg.deserialize(dat)



# ------------------------------------------------------------------------------
# Translation to standard interpolation language
# ------------------------------------------------------------------------------

# get the interpolation/hierarchical coefficients
C = asg.coefficients(G)

# evaluate the basis function (as a basis vector) at a point
x = rand(D)
B = asg.basis(x, G)

# (manually) evaluate the interpolant at the point
sum(B .* C)

# evaluate the basis function at multiple points
X = rand(10, D)  # 10 points
B = asg.basis(X, G)

# (default) evaluate the basis function at all the grid nodes
B = asg.basis(G)
@assert size(B) == (length(G), length(G))

# construct: hierarchization and de-hierarchization matrices
H = asg.hierarchization_matrix(G)
E = asg.dehierarchization_matrix(G) # equivalent to `asg.basis(G)`



# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

# default: dimensional check applies, no extrapolation
x = rand(D)
G(x) 

# no dimensional check, (linear) extrapolation allowed
x = rand(D) .+ 1.0
G(
    x, 
    safe = false, 
    extrapolate = true
)



# ------------------------------------------------------------------------------
# A 2-D visualization example
# ------------------------------------------------------------------------------

# mimics: exponential utility function which needs adpation as x --> 0
f2fit(X) = -1 / (0.1 + sum(X))

G = asg.SparseGridInterpolant(2)
asg.train!(
    G, 
    f2fit, 
    4, 
    (3,8),
    verbose = true, # display training progress
)
asg.adapt!(
    G, f2fit, 20,
    verbose = true,
    tol     = 1e-5,
    toltype = :absolute,
    parallel = false,
)


import Plots as plt

# fig: approximation errors
fig = plt.surface(
    LinRange(G, 1, 100),
    LinRange(G, 2, 100),
    (x,y) -> G([x,y]) - f2fit([x,y]),
    camera = (-30,30),
    alpha = 0.5,
    colorbar = false,
    legend = false,
)
fig

# fig: grid nodes distribution
fig = plt.surface(
    LinRange(G, 1, 100),
    LinRange(G, 2, 100),
    (x,y) -> G([x,y]),
    camera = (-30,30),
    alpha = 0.5,
    colorbar = false,
    legend = false,
)
plt.scatter!(
    fig,
    (stack(G) |> eachcol)...,
    G.fvals,
)
fig
```
"""
mutable struct SparseGridInterpolant{D} <: AbstractSparseGridInterpolant{D}
    
    # domain information
    lb::NTuple{D, Float64}  # lower bound of the domain
    ub::NTuple{D, Float64}  # upper bound of the domain

    # nodes information (defined in [0,1]^D hypercube)
    levels ::Matrix{Int}  # levels of the nodes, N * D
    indices::Matrix{Int}  # indices of the nodes, N * D
    depths ::Vector{Int}  # depth of the nodes (in the hierarchy/tree), N

    # coefficients information
    fvals  ::Vector{Float64}  # function values at the nodes, N
    coefs  ::Vector{Float64}  # interpolation/hierarchical coefficients, N

    function SparseGridInterpolant(
        xdim::Int ;
        lb::AbsVecTup = zeros(Float64, xdim),
        ub::AbsVecTup = ones(Float64, xdim),
    )
        @assert xdim > 0 "Dimension must be positive."
        @assert length(lb) == xdim "Lower bound must match dimension."
        @assert length(ub) == xdim "Upper bound must match dimension."
        @assert all(ub .> lb) "Upper bound must be greater than lower bound."
        @assert all(isfinite, lb) "Lower bound must be finite."
        @assert all(isfinite, ub) "Upper bound must be finite."
        new{xdim}(
            lb |> NTuple{xdim, Float64},
            ub |> NTuple{xdim, Float64},
            
            Matrix{Int}(undef, 0, xdim),  # empty levels matrix
            Matrix{Int}(undef, 0, xdim),  # empty indices matrix
            Int[],                        # empty depths vector

            Float64[],                    # empty function values vector
            Float64[]                     # empty coefficients vector
        )
    end
end # SparseGridInterpolant{D}
# ------------------------------------------------------------------------------
function Base.show(io::IO, G::SparseGridInterpolant{D}) where {D}
    @printf(io, "SparseGridInterpolant f(x): R^%d --> R\n", D)
    @printf(io, "#nodes: %d\n", length(G.fvals))
    println(io, "Domain:")
    for d in 1:D
        @printf(io, "  x[%d] in [%.2f, %.2f]\n", d, G.lb[d], G.ub[d])
    end
    return nothing
end
# ------------------------------------------------------------------------------
function Base.length(G::SparseGridInterpolant{D}) where {D}
    return length(G.fvals)
end
# ------------------------------------------------------------------------------
function Base.ndims(G::SparseGridInterpolant{D}) where {D}
    return D
end
# ------------------------------------------------------------------------------
function Base.size(G::SparseGridInterpolant{D}) where {D}
    return (length(G.fvals), D)
end
# ------------------------------------------------------------------------------
function Base.in(x::Point, G::SparseGridInterpolant{D}) where {D}
    @assert length(x) == D "Input point must match the dimension of the grid."
    return all(x .>= G.lb) && all(x .<= G.ub)
end
# ------------------------------------------------------------------------------
function Base.clamp(x::Point, G::SparseGridInterpolant{D}) where {D}
    return clamp.(x, G.lb, G.ub) 
end
# ------------------------------------------------------------------------------
function Base.rand(G::SparseGridInterpolant{D}) where {D}
    # generate a random point in the domain of the grid
    return rand(Float64, D) .* (G.ub .- G.lb) .+ G.lb
end
# ------------------------------------------------------------------------------
function Base.rand(G::SparseGridInterpolant{D}, n::Int) where {D}
    # generate `n` random points in the domain of the grid
    # returns a matrix of size `D * n`
    return rand(Float64, D, n) .* (G.ub .- G.lb) .+ G.lb
end
# ------------------------------------------------------------------------------
function Base.LinRange(G::SparseGridInterpolant{D}, d::Int, n::Int) where {D}
    @assert 1 <= d <= D "Dimension index must be in [1, D]."
    return LinRange(G.lb[d], G.ub[d], n)
end
# ------------------------------------------------------------------------------
"""
    Base.stack(G::SparseGridInterpolant{D})::Matrix{Float64} where {D}

Stack the grid nodes into a matrix of size `N * D`, where `N` is the number of
nodes and `D` is the dimension of the grid. Each row corresponds to a node in
the grid, and each column corresponds to a dimension.
"""
function Base.stack(G::SparseGridInterpolant{D})::Matrix{Float64} where {D}
    if G.levels |> isempty
        return Matrix{Float64}(undef, 0, D)  # return an empty matrix
    else
        return stack(
            [
                get_x(Ls, Is, lb = G.lb, ub = G.ub)
                for (Ls, Is) in zip(G.levels |> eachrow, G.indices |> eachrow)
            ],
            dims = 1
        )
    end
end




#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SECTION: Interface methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
"""
    maxlevels(G::SparseGridInterpolant{D})::Vector{Int} where {D}

Return the maximum (hierarchical) levels of the sparse grid interpolant `G` 
along each dimension.
"""
function maxlevels(G::SparseGridInterpolant{D})::Vector{Int} where {D}
    if isempty(G.levels)
        return zeros(Int, D)  # return a zero vector if no nodes
    else
        return maximum(G.levels, dims = 1) |> vec
    end
end
# ------------------------------------------------------------------------------
"""
    coefficients(G::SparseGridInterpolant{D})::Vector{Float64} where {D}

Return the interpolation/hierarchical coefficients as a vector
"""
function coefficients(G::SparseGridInterpolant{D}) where {D}
    return G.coefs
end
# ------------------------------------------------------------------------------
"""
    basis(
        x::AbstractVector,
        G::SparseGridInterpolant{D} ;

        dropzeros::Bool = true,
        atol     ::Float64  = 1e-16
    )::SparseVector{Float64,Int} where {D}

Create a `SparseVector{Float64,Int}` of the basis vector at the point `x` using
the ASG `G`. The `x` can be any point in the domain but not necessarily in the
grid `G`.

Returns a sparse vector of length `length(G)`, which corresponds to a row in a
basis matrix of many `x` points.
"""
function basis(
    x::AbstractVector,
    G::SparseGridInterpolant{D} ;

    dropzeros::Bool = true,
    atol     ::Float64  = 1e-16
)::SparseVector{Float64,Int} where {D}

    # scale the point `x` to the unit hypercube [0,1]^D
    x01 = scale(x, G.lb, G.ub)

    vals = G |> length |> spzeros
    for (xj,Lj,Ij) in zip(x01, G.levels |> eachcol, G.indices |> eachcol)
        vals .*= ϕ.(xj, Lj, Ij)
    end
    
    if dropzeros
        vals[vals .< atol] .= 0.0
    end
    return vals |> SparseArrays.dropzeros
end
# ------------------------------------------------------------------------------
"""
    basis(
        Xs::AbstractMatrix,
        G::SparseGridInterpolant{D} ;

        safe     ::Bool = true,
        dropzeros::Bool = true,
        atol     ::Float64  = 1e-16
    )::SparseMatrixCSC{Float64,Int} where {D}

Create a `SparseMatrixCSC{Float64,Int}` of the basis matrix at the points `X`
using the interpolant `G`. The `X` can be any points in the domain but not 
necessarily in the grid `G`.

# Arguments
- `X::AbstractMatrix`: The points at which the basis matrix is to be calculated.
The matrix should have `D` columns, where each row is a point.
- `G::SparseGridInterpolant{D}`: The interpolant.
- `safe::Bool`: If `true`, then the basis functions are evaluated in a safe
  manner, i.e., checking the validity of the input.
- `dropzeros::Bool`: Drop the zero weights. Default is `true`.
- `atol::Float64`: Abs tolerance to consider a weight as zero. Default `1e-16`.

# Returns
- A sparse matrix of size `(size(X,1), length(G))` where each row corresponds
to the basis vector at the corresponding point in `X`.
"""
function basis(
    Xs::AbstractMatrix,
    G::SparseGridInterpolant{D} ;

    safe     ::Bool = true,
    dropzeros::Bool = true,
    atol     ::Float64  = 1e-16
)::SparseMatrixCSC{Float64,Int} where {D}
    return stack(
        [
            basis(
                x, G, 
                safe = safe,
                dropzeros = dropzeros,
                atol = atol
            )
            for x in Xs |> eachrow
        ],
        dims = 1
    )
end
# ------------------------------------------------------------------------------
"""
    basis(
        G::SparseGridInterpolant{D} ;
        safe     ::Bool = true,
        dropzeros::Bool = true,
        atol     ::Float64  = 1e-16
    )::SparseMatrixCSC{Float64,Int} where {D}

Create a `SparseMatrixCSC{Float64,Int}` of the basis matrix at all the grid 
nodes using the interpolant `G`. The basis matrix is a matrix where each row
corresponds to the basis vector at the corresponding node in `G`.

## Notes
- The on-grid basis matrix is NOT an identity matrix like the regular piecewise
linear interpolation.
"""
function basis(
    G::SparseGridInterpolant{D} ;
    safe     ::Bool = true,
    dropzeros::Bool = true,
    atol     ::Float64  = 1e-16
)::SparseMatrixCSC{Float64,Int} where {D}
    if isempty(G.levels)
        return SparseMatrixCSC{Float64,Int}(undef, 0, length(G))  # empty matrix
    else
        return basis(
            G |> stack,
            G,
            safe = safe,
            dropzeros = dropzeros,
            atol = atol
        )
    end
end
# ------------------------------------------------------------------------------
"""
    dehierarchization_matrix(
        G::SparseGridInterpolant{D} ;
        safe     ::Bool = true,
        dropzeros::Bool = true,
        atol     ::Float64  = 1e-16
    )::SparseMatrixCSC{Float64,Int} where {D}

Get the de-hierarchization matrix `E` of the ASG `G`. The de-hierarchization ma-
trix is a sparse matrix that can be applied to the hierarchical coefficients of
the ASG to get the nodal values (interpolated function values at nodes).

The de-hierarchization matrix is identical to the basis matrix. This terminology
is used in the literature and usually denoted as `E` matrix.
"""
function dehierarchization_matrix(
    G::SparseGridInterpolant{D} ;
    safe     ::Bool = true,
    dropzeros::Bool = true,
    atol     ::Float64  = 1e-16
)::SparseMatrixCSC{Float64,Int} where {D}
    return basis(
        G, 
        safe = safe, 
        dropzeros = dropzeros, 
        atol = atol
    )
end
# ------------------------------------------------------------------------------
"""
    hierarchization_matrix(
        G::SparseGridInterpolant{D} ;
        safe     ::Bool = true,
        dropzeros::Bool = true,
        atol     ::Float64  = 1e-16
    )::SparseMatrixCSC{Float64,Int} where {D}

Get the hierarchization matrix `E` of the ASG `G`. The hierarchization matrix is
a sparse matrix that can be applied to the nodal coefficients.

However, we recommand to use "H * nodal_values = E \\ nodal_values" instead of
computing `H` explicitly (if possible).

The hierarchization matrix is basically the inverse of the 
de-hierarchization matrix. The literature usually denotes it as `H` matrix.

## Notes
- IMPORTANT: The `lu()` for sparse matrix in Julia behaves differently from the
`lu()` for dense matrix, so we cannot directly use `E^{-1}=U^{-1}*L^{-1}`. There
is no solution yet. (Jul 2025)
"""
function hierarchization_matrix(
    G::SparseGridInterpolant{D} ;
    safe     ::Bool = true,
    dropzeros::Bool = true,
    atol     ::Float64  = 1e-16
)::SparseMatrixCSC{Float64,Int} where {D}
    E = dehierarchization_matrix(
        G, 
        safe = safe, 
        dropzeros = dropzeros, 
        atol = atol
    )
    return sparse(E \ Matrix{Float64}(SparseArrays.I, size(E)...))
end









#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SECTION: Evaluation API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
"""
    _interpolate(
        G   ::SparseGridInterpolant{D}, 
        x   ::Point ; 
        safe::Bool = true
    )

Evalutes the sparse grid interpolant `G` at the point `x`. The point `x` must be
in the domain of the grid, otherwise, the calling will return a diminishing 0.

This method is a private helper function.
"""
function _interpolate(
    G   ::SparseGridInterpolant{D}, 
    x   ::Point ; 
    safe::Bool = true
) where {D}
    # step: affine the point `x` to the unit hypercube [0,1]^D
    x01 = scale(x, G.lb, G.ub)

    # coefficients are pre-allocated in the grid
    vals = G.coefs |> copy
    for (xj,Lj,Ij) in zip(x01, G.levels |> eachcol, G.indices |> eachcol)
        vals .*= ϕ.(xj, Lj, Ij)
    end
    return sum(vals)
end
# ------------------------------------------------------------------------------
function _interpolate_if(
    G    ::SparseGridInterpolant{D}, 
    x    ::Point,
    idx  ::Union{Vector{Bool}, BitVector, Vector{Int}} ;
    safe ::Bool = true
) where {D}
    # internal method for the training process. not for public use.

    x01  = scale(x, G.lb, G.ub)
    
    vals = G.coefs[idx] |> copy
    for (xj,Lj,Ij) in zip(
        x01, 
        G.levels[idx,:] |> eachcol, 
        G.indices[idx,:] |> eachcol
    )
        vals .*= ϕ.(xj, Lj, Ij)
    end
    return sum(vals)
end
# ------------------------------------------------------------------------------
function _interpolate_until_level(
    G          ::SparseGridInterpolant{D}, 
    x          ::Point,
    until_level::Int ; 
    safe       ::Bool = true
) where {D}
    # internal method for the training process. not for public use.

    return _interpolate_if(
        G, 
        x, 
        G.depths .<= until_level,
        safe = safe
    )
end
# ------------------------------------------------------------------------------
"""
    _extrapolate(
        G   ::SparseGridInterpolant{D}, 
        x   ::Point ; 
        safe::Bool = true
    ) where {D}

Linearly extrapolates the sparse grid interpolant `G` to a point `x` that is
(partially or fully) outside the domain. Using 1st order Taylor expansion to
approximate the value.

This method is a private helper function. Depends on `_interpolate()`.
"""
function _extrapolate(
    G   ::SparseGridInterpolant{D}, 
    x   ::Point ; 
    safe::Bool = true
) where {D}

    # NOTES: the scaling operation changes the approximation for computing the
    #        partial derivatives

    # scale the point `x` to the unit hypercube [0,1]^D
    x01   = scale(x, G.lb, G.ub)
    xgaps = G.ub .- G.lb

    fval = 0.0

    # step: get the nearest boundary point & eval there
    xbnd  = clamp.(x01, 0.0, 1.0)
    fval += _interpolate(G, xbnd, safe = safe)

    # step: get the maximum hierarchical levels
    maxLvls = maxlevels(G)

    # step: evaluate the Taylor expansion at `x01` by expanding the function at
    #       the boundary point `xbnd`
    for j in 1:D
        (0.0 <= x01[j] <= 1.0) && continue # skip the interior & boundary points

        ΔxOut  = x01[j] - xbnd[j]
        ΔxMesh = (maxLvls[j] |> ghost_distance) * xgaps[j]

        flag_right = x01[j] > 1.0

        # toward interior along dimension j, get a neighbor point
        xnbr     = copy(xbnd)
        xnbr[j] -= flag_right ? ΔxMesh : -ΔxMesh
        fnbr     = _interpolate(G, xnbr, safe = safe)

        ∂f∂xj = (flag_right ? (fval - fnbr) : (fnbr - fval)) / ΔxMesh

        fval += ∂f∂xj * ΔxOut
    end # j
    return fval
end
# ------------------------------------------------------------------------------
"""
    (G::SparseGridInterpolant{D})(x::Point ; safe::Bool = true)

Evalutes the sparse grid interpolant `G` at the point `x`. The point `x` must be
in the domain of the grid, otherwise, the calling will return a diminishing 0.

## Notes
"""
function (G::SparseGridInterpolant{D})(
    x           ::Point ; 
    safe        ::Bool = true,
    extrapolate ::Bool = false,
) where {D}
    return if extrapolate
        if x ∈ G
            __extrapolate(G, x, safe = safe)
        else
            _interpolate(G, x, safe = safe)
        end
    else
        _interpolate(G, x, safe = safe)
    end
end # eval


