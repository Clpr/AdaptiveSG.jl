# EXTRAPOLATION BY TAYLOR SERIES
# ------------------------------------------------------------------------------
export isoutside
export extrapolate

# ------------------------------------------------------------------------------
"""
    isoutside(x::AbstractVector)::Bool

Returns `true` if any element of the input vector `x` is outside the rectangle
`[0,1]^d`, and `false` otherwise.

This function cannot identify the case of NaNs but works with +/-Inf values.
"""
function isoutside(x::AbstractVector)::Bool
    for xi in x
        if (xi < 0.0) || (xi > 1.0)
            return true
        end
    end
    return false
end # isoutside
# ------------------------------------------------------------------------------
"""
    isoutside(x::AbstractVector, nzer::Normalizer)::Bool

Returns `true` if any element of the input vector `x` is outside the rectangle
`[lb,ub]^d`, where `lb` and `ub` are the lower and upper bounds defined in the
normalizer `nzer`.

This function cannot identify the case of NaNs but works with +/-Inf values.
"""
function isoutside(x::AbstractVector, nzer::Normalizer)::Bool
    for (xi, lbi, ubi) in zip(x, nzer.lb, nzer.ub)
        if (xi < lbi) || (xi > ubi)
            return true
        end
    end
    return false
end # isoutside



# ------------------------------------------------------------------------------
"""
    extrapolate(
        G  ::AbstractSparseGrid,
        x01::AbstractVector{Float64}
    )::Float64

Linearly extrapolates the ASG interpolant `G` to a point `x01` that is outside
the domain `[0,1]^d`. The extrapolation is done by computing the 1st order
Taylor expansion of the interpolant at the nearest boundary point.
"""
function extrapolate(
    G  ::AbstractSparseGrid{d},
    x01::AbstractVector{Float64}
)::Float64 where {d}

    fval = 0.0

    # step: clamp to get a nearest boundary/vertex point
    xbnd::Vector{Float64} = clamp.(x01, 0.0, 1.0) |> collect
    fval += evaluate(G, xbnd)

    # step: evaluate the Taylor expansion at `x01` by expanding the function at
    #       the boundary point `xbnd`
    for j in 1:d
        (0.0 <= x01[j] <= 1.0) && continue # skip the interior & boundary points

        ΔxOut  = x01[j] - xbnd[j]
        ΔxMesh = G.max_levels[j] |> get_ghost_stepsize

        flag_right = x01[j] > 1.0

        # toward interior along dimension j, get a neighbor point
        xnbr     = copy(xbnd)
        xnbr[j] -= flag_right ? ΔxMesh : -ΔxMesh
        fnbr     = evaluate(G, xnbr)

        ∂f∂xj = (flag_right ? (fval - fnbr) : (fnbr - fval)) / ΔxMesh

        fval += ∂f∂xj * ΔxOut
    end # j
    return fval
end # extrapolate
# ------------------------------------------------------------------------------
"""
    extrapolate(
        G    ::AbstractSparseGrid,
        x    ::AbstractVector{Float64},
        nzer ::Normalizer
    )::Float64

Linearly extrapolates the ASG interpolant `G` to a point `x` that is outside
the domain `[lb,ub]^d`, where `lb` and `ub` are the lower and upper bounds
defined in the normalizer `nzer`.
"""
function extrapolate(
    G    ::AbstractSparseGrid{d},
    x    ::AbstractVector{Float64},
    nzer ::Normalizer{d} ;
)::Float64 where {d}
    # NOTES: because the scaling changes the approximation for computing the
    #        partial derivatives, I repeat the same but slightly different 
    #        code here without introducing more abstractions.

    fval = 0.0

    xbnd::Vector{Float64} = clamp(x, nzer) |> collect
    fval += evaluate(G, xbnd)

    for j in 1:d
        (nzer.lb[j] <= x[j] <= nzer.ub[j]) && continue

        ΔxOut  = x[j] - xbnd[j]
        ΔxMesh = denormalize_dist_along(
            G.max_levels[j] |> get_ghost_stepsize,
            nzer, j
        )

        flag_right = x[j] > nzer.ub[j]

        xnbr     = copy(xbnd)
        xnbr[j] -= flag_right ? ΔxMesh : -ΔxMesh
        fnbr     = evaluate(G, xnbr)

        ∂f∂xj = (flag_right ? (fval - fnbr) : (fnbr - fval)) / ΔxMesh

        fval += ∂f∂xj * ΔxOut
    end # j
    return fval    
end # extrapolate






