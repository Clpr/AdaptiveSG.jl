# Numerical integration of ASG/RSG interpolants.
# ==============================================================================
export integrate


# ------------------------------------------------------------------------------
"""
    integrate(G, xref, dim; xlim=(0.0, 1.0), nsamples=10, weight=ones(nsamples))

Numerical integration of ASG/RSG interpolants, using composite trapezoidal rule.
This is 1-dim partial integral along the `dim`-th dimension of the interpolant.

# Arguments
- `G::AbstractSparseGrid{d}`: ASG/RSG interpolant
- `xref::AbstractVector{Float64}`: reference point in [0,1]^d
- `dim::Int`: dimension to integrate
- `xlim::NTuple{2,Float64}`: integration limits, default=(0.0, 1.0)
- `nsamples::Int`: number of samples, default=10
- `weight::AbstractVector{Float64}`: weights for each sample

# Returns
- `Float64`: integral value

# Notes
- The `xref` is in [0,1]^d, and the integration is along the `dim`-th dimension.
It means that the `dim`-th coordinate of `xref` is perturbed by the integration
variable, while the other coordinates are fixed.
"""
function integrate(
    G         ::AbstractSparseGrid{d},
    xref      ::AbstractVector{Float64},
    dim       ::Int ;
    xlim      ::NTuple{2,Float64} = (0.0, 1.0),
    nsamples  ::Int = 10,
    weight    ::AbstractVector{Float64} = ones(nsamples)
)::Float64 where {d}
    if (xlim[1] >= xlim[2])
        throw(ArgumentError("xlim[1] must be < xlim[2]"))
    end
    if (xlim[1] < 0.0) || (xlim[2] > 1.0)
        throw(ArgumentError("xlim must be within [0,1]"))
    end

    # note: xref is in [0,1]^d
    f2int(xj) = evaluate(
        G,
        Float64[ xref[1:dim-1]; xj; xref[dim+1:end] ]
    )

    Δx = (xlim[2] - xlim[1]) / (nsamples - 1)
    Zs = f2int.(LinRange(xlim[1], xlim[2], nsamples)) .* weight
    vint = sum(Zs) - 0.5 * (Zs[1] + Zs[end])

    return vint * Δx
end # integrate()


# ------------------------------------------------------------------------------
"""
    integrate(
        G, xref, dim, nzer; 
        xlim=(0.0, 1.0), nsamples=10, weight=ones(nsamples)
    )

Numerical integration of ASG/RSG interpolants, using composite trapezoidal rule.
But this version uses a normalizer to allow `xref` and `xlim` to be in the 
original domain (hyper rectangle).
"""
function integrate(
    G         ::AbstractSparseGrid{d},
    xref      ::AbstractVector{Float64},
    dim       ::Int,
    nzer      ::Normalizer{d} ;
    xlim      ::NTuple{2,Float64} = (nzer.lb[dim], nzer.ub[dim]),
    nsamples  ::Int = 10,
    weight    ::AbstractVector{Float64} = ones(nsamples)
)::Float64 where {d}
    xref01 = normalize(xref, nzer)
    xlim01 = (normalize(xlim[1], nzer, dim), normalize(xlim[2], nzer, dim))
    vint01 = integrate(G,xref01,dim,xlim=xlim01,nsamples=nsamples,weight=weight)
    # note: there is a scaling after integrating over the normalized domain
    return (xlim[2] - xlim[1]) * vint01
end # integrate()
















