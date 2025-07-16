# API for standard interpolation terminologies in numerical analysis theories.

export interpcoef
export basis_matrix
export get_dehierarchization_matrix
export get_hierarchization_matrix


# ------------------------------------------------------------------------------
"""
    interpcoef(G::AbstractSparseGrid{d})::Vector{Float64} where d

Get the interpolation coefficient vector, equivalent to the hierarchical coeffi-
cient vector, of the sparse grid `G`. Returns a vector of length `length(G)`.

This method is wrapped for users who are familiar to or work in the context of
typical interplation in mathematics, where the interpolation coefficient is the
quantity of interest. The method is equivalent to `vectorize_hierarchical`.

To get the basis matrix at nodes, check `basis_matrix()` method which equivalent
to `get_dehierarchization_matrix()`; to evaluate the basis matrix at arbitrary 
point(s), check `basis_matrix()` method as well.
"""
function interpcoef(G::AbstractSparseGrid{d})::Vector{Float64} where {d}
    return vectorize_hierarchical(G)
end


# ------------------------------------------------------------------------------
"""
    basis_matrix(
        x::AbstractVector,
        G::AbstractSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64  = 1e-16
    )::SparseVector{Float64,Int} where d

Create a `SparseVector{Float64,Int}` of the basis vector at the point `x` using
the ASG `G`. The `x` can be any point in the domain but not necessarily in the
grid `G`. Notice: `x` in `[0,1]^d`. If you need scaling, do it before calling
this method.
"""
function basis_matrix(
    x::AbstractVector,
    G::AbstractSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64  = 1e-16
)::SparseVector{Float64,Int} where d
    Is = Int[]; Vs = Float64[]
    for (i, p) in enumerate(keys(G.nv))
        w = ϕ_unsafe(x, p)
        if dropzeros && isapprox(w, 0.0, atol = atol); continue; end
        push!(Is, i)
        push!(Vs, w)
    end
    return sparsevec(Is, Vs, length(G))
end # basis_matrix()


# ------------------------------------------------------------------------------
"""
    basis_matrix(
        p::Node{d},
        G::AbstractSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64  = 1e-16
    )::SparseVector{Float64,Int} where d

Alias for `basis_matrix(get_x(p), G)`.
"""
function basis_matrix(
    p::Node{d}, 
    G::AbstractSparseGrid{d} ; 
    dropzeros::Bool = true, 
    atol::Float64 = 1e-16
)::SparseVector{Float64,Int} where d
    return basis_matrix(get_x(p), G, dropzeros = dropzeros, atol = atol)
end # basis_matrix()



# ------------------------------------------------------------------------------
"""
    basis_matrix(
        X::AbstractMatrix,
        G::AbstractSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64  = 1e-16
    )::SparseMatrixCSC{Float64,Int} where d

Create a `SparseMatrixCSC{Float64,Int}` of the basis matrix at the points `X`
using the ASG `G`. The `X` can be any points in the domain but not necessarily
in the grid `G`. Notice: `X[i,:]` in `[0,1]^d`. If you need scaling, do it 
before calling this method.

# Arguments
- `X::AbstractMatrix`: The points at which the basis matrix is to be calculated.
The matrix should have `d` columns, where each row is a point.
- `G::AdaptiveSparseGrid{d}`: The ASG.
- `dropzeros::Bool`: Drop the zero weights. Default is `true`.
- `atol::Float64`: Abs tolerance to consider a weight as zero. Default `1e-16`.

# Returns
- A sparse matrix of size `(size(X,1), length(G))` where each row corresponds
to the basis vector at the corresponding point in `X`.
"""
function basis_matrix(
    X::AbstractMatrix,
    G::AbstractSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64  = 1e-16
)::SparseMatrixCSC{Float64,Int} where d
    Is = Int[]; Js = Int[]; Vs = Float64[]
    for (j, p) in G.nv |> keys |> enumerate
        for (i, x) in X |> eachrow |> enumerate
            w = ϕ_unsafe(x, p)
            if dropzeros && isapprox(w, 0.0, atol = atol); continue; end
            push!(Is, i)
            push!(Js, j)
            push!(Vs, w)
        end
    end # (j,p)
    return sparse(Is, Js, Vs, size(X,1), length(G))
end # basis_matrix()


# ------------------------------------------------------------------------------
"""
    basis_matrix(
        G::AbstractSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64  = 1e-16
    )::SparseMatrixCSC{Float64,Int} where d

Create a `SparseMatrixCSC{Float64,Int}` of the basis matrix at the nodes of the
ASG `G`. The basis matrix is a matrix where each row corresponds to the basis
vector at the corresponding node in the grid.

Alias for `basis_matrix(vectorize_x(G), G)`; 
also alias for `get_hierarchization_matrix()`.
"""
function basis_matrix(
    G::AbstractSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64  = 1e-16
) where d
    return basis_matrix(vectorize_x(G), G, dropzeros = dropzeros, atol = atol)
end # basis_matrix()


# ------------------------------------------------------------------------------
"""
    get_dehierarchization_matrix(
        G::AdaptiveSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64   = 1E-16
    )::SparseMatrixCSC{Float64} where d

Get the de-hierarchization matrix `E` of the ASG `G`. The de-hierarchization ma-
trix is a sparse matrix that can be applied to the hierarchical coefficients of
the ASG to get the nodal values (interpolated function values at nodes).

## Notes
- The `dropzeros` and `atol` arguments are used to avoid storing the zero values
in the matrix. The zero values are dropped if `dropzeros = true` and the values
are considered zero if `abs(value) < atol`.
- We drop zeros during the matrix construction rather than call `dropzeros` then
"""
function get_dehierarchization_matrix(
    G::AdaptiveSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64   = 1E-16
)::SparseMatrixCSC{Float64,Int} where d
    return basis_matrix(G, dropzeros = dropzeros, atol = atol)
end # get_dehierarchization_matrix()


# ------------------------------------------------------------------------------
"""
    get_hierarchization_matrix(
        G::AdaptiveSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64   = 1E-16
    )::SparseMatrixCSC{Float64} where d

Get the hierarchization matrix `E` of the ASG `G`. The hierarchization matrix is
a sparse matrix that can be applied to the nodal coefficients.

However, we recommand to use "H * nodal_values = E \\ nodal_values" instead of
computing `H` explicitly (if possible).

## Notes
- IMPORTANT: The `lu()` for sparse matrix in Julia behaves differently from the
`lu()` for dense matrix, so we cannot directly use `E^{-1}=U^{-1}*L^{-1}`. There
is no solution yet.
"""
function get_hierarchization_matrix(
    G::AdaptiveSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64   = 1E-16
)::SparseMatrixCSC{Float64} where d
    E = get_dehierarchization_matrix(G, dropzeros = dropzeros, atol = atol)
    return sparse(E \ Matrix{Float64}(I, size(E)...))
end # get_hierarchization_matrix()
