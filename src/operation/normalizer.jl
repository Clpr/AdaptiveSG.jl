export normalize, denormalize


# ------------------------------------------------------------------------------
"""
    normalize(
        x::AbstractVector{Float64}, 
        nzer::Normalizer{d}
    )::SVector{d,Float64} where d

Normalize the input vector `x` to the domain `[0,1]^d` using the `Normalizer{d}`
"""
function normalize(
    x::AbstractVector{Float64}, 
    nzer::Normalizer{d}
)::SVector{d,Float64} where d
    if length(x) != d; throw(ArgumentError("length(x) != d")); end
    return SVector{d,Float64}([
        (x[i] - nzer.lb[i]) / nzer.gap[i]
        for i in 1:length(x)
    ]...)
end # normalize()


# ------------------------------------------------------------------------------
"""
    normalize(x::Float64, nzer::Normalizer{d}, dims::Int)::Float64 where d

Normalize the input scalar `x` to the domain `[0,1]` using the `Normalizer{d}`,
where `dims` is the dimension of the scalar.
"""
function normalize(x::Float64, nzer::Normalizer{d}, dims::Int)::Float64 where d
    return (x - nzer.lb[dims]) / nzer.gap[dims]
end # normalize()


# ------------------------------------------------------------------------------
"""
    denormalize(
        x::AbstractVector{Float64}, 
        nzer::Normalizer{d}
    )::SVector{d,Float64} where d

Denormalize the input vector `x` from `[0,1]^d` to its original domain using the
`Normalizer{d}`.
"""
function denormalize(
    x::AbstractVector{Float64}, 
    nzer::Normalizer{d}
)::SVector{d,Float64} where d
    if length(x) != d; throw(ArgumentError("length(x) != d")); end
    return SVector{d,Float64}([
        x[i] * nzer.gap[i] + nzer.lb[i] 
        for i in 1:length(x)
    ]...)
end # denormalize()


# ------------------------------------------------------------------------------
"""
    denormalize(x::Float64, nzer::Normalizer{d}, dims::Int)::Float64 where d

Denormalize the input scalar `x` from `[0,1]` to its original domain using the
`Normalizer{d}`, where `dims` is the dimension of the scalar.
"""
function denormalize(x::Float64, nzer::Normalizer{d},dims::Int)::Float64 where d
    return x * nzer.gap[dims] + nzer.lb[dims]
end # denormalize()

