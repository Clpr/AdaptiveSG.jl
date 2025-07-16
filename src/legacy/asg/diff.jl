# Differentiation: Jacobian & Hessian
# ------------------------------------------------------------------------------
export gradient


# ------------------------------------------------------------------------------
function _gradient_bearbone!(
    ∇G ::Vector{Float64},
    x2 ::Vector{Float64},
    G  ::AbstractSparseGrid{d},
    Δx ::Vector{Float64},
    x  ::AbstractVector,
    mode         ::Symbol,
    extrapolation::Bool
)::Nothing where {d}
    for j in 1:d

        Δxr = mode == :backward ? 0.0 : Δx[j]
        Δxl = mode == :forward  ? 0.0 : Δx[j]

        x2[j] = x[j] + Δxr
        fxr = evaluate(G, x2, extrapolation = extrapolation)
        x2[j] = x[j] - Δxl
        fxl = evaluate(G, x2, extrapolation = extrapolation)

        ∇G[j] = (fxr - fxl) / (Δxr + Δxl)

    end # j
    return nothing
end # _gradient_bearbone!
# ------------------------------------------------------------------------------
"""
    gradient(
        G::AbstractSparseGrid{d}, 
        x::AbstractVector; 
        mode::Symbol = :central, 
        extrapolation::Bool = false
    )::Vector{Float64}

Compute the gradient of the ASG/RSG interpolant at a point `x` in [0,1]^d using 
finite differences. The `mode` argument can be one of `:central`, `:forward`, or
`:backward`. The `extrapolation` argument is used to determine whether to
linearly extrapolate the interpolant outside the domain of the grid. Shutting
down this option if `x` is strictly interior to improve performance.

## Notes
- An alternative implementation is to hard-code the basis function evaluations
as loops for performance (about 2x faster and 7x less allocs). However, this 
implementation require the point `x` and its neighbors must be all in the domain
of the grid. This will introduce extra checks and dispatches. I mark it here for
future optimization.
"""
function gradient(
    G   ::AbstractSparseGrid{d}, 
    x   ::AbstractVector ;
    mode         ::Symbol = :central,
    extrapolation::Bool   = false
)::Vector{Float64} where {d}

    if mode ∉ (:central, :forward, :backward)
        throw(ArgumentError("mode = $mode not supported"))
    end
    ∇G = Vector{Float64}(undef, d)
    x2 = Vector{Float64}(x)
    Δx = get_ghost_stepsize(G)
    _gradient_bearbone!(∇G, x2, G, Δx, x, mode, extrapolation)
    return ∇G
end # gradient
# ------------------------------------------------------------------------------
"""
    gradient(
        G::AbstractSparseGrid{d}, 
        x::AbstractVector,
        nzer::Normalizer{d} ;
        mode::Symbol = :central, 
        extrapolation::Bool = false
    )::Vector{Float64}

The scaling version of `gradient()`. The `x` is in the original domain of the
function, and `nzer` is the normalizer used to train the grid.
"""
function gradient(
    G   ::AbstractSparseGrid{d}, 
    x   ::AbstractVector,
    nzer::Normalizer{d} ;
    mode         ::Symbol = :central,
    extrapolation::Bool   = false
)::Vector{Float64} where {d}

    if mode ∉ (:central, :forward, :backward)
        throw(ArgumentError("mode = $mode not supported"))
    end
    ∇G   = Vector{Float64}(undef, d)
    x2   = Vector{Float64}(x)
    Δx01 = get_ghost_stepsize(G)
    _gradient_bearbone!(∇G, x2, G, Δx01, normalize(x,nzer), mode, extrapolation)

    # scaling back
    ∇G ./= nzer.gap
    return ∇G
end # gradient


