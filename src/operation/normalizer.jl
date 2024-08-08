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

## Notes
- This method is for affining a **point** `x[j]` along `j` dimension, but not 
the distance between any two points. To affine a distance `z:=x2[j]-x1[j]`, one
should directly do `(x2[j]-x1[j]) / nzer.gap[j]`. The intercept (lower bound) is
cancelled out in the subtraction.
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

## Notes
- This method is for affining a **point** `x[j]` along `j` dimension, but not 
the distance between any two points. To affine a distance `z:=x2[j]-x1[j]`, one
should directly do `(x2[j]-x1[j]) * nzer.gap[j]`. The intercept (lower bound) is
cancelled out in the subtraction.
"""
function denormalize(x::Float64, nzer::Normalizer{d},dims::Int)::Float64 where d
    return x * nzer.gap[dims] + nzer.lb[dims]
end # denormalize()


# ------------------------------------------------------------------------------
"""
    normalize_dist_along(
        dist_original_space::Float64, 
        nzer::Normalizer{d}, 
        dims::Int
    )::Float64 where d

Normalize the input distance `dist` along the `dims` dimension to the domain 
`[0,1]`. This is different from normalizing a point.
"""
function normalize_dist_along(
    dist_original_space::Float64, 
    nzer::Normalizer{d}, 
    dims::Int
)::Float64 where d
    return dist_original_space / nzer.gap[dims]
end # normalize_dist_along()


# ------------------------------------------------------------------------------
"""
    denormalize_dist_along(
        dist_normalized::Float64, 
        nzer::Normalizer{d}, 
        dims::Int
    )::Float64 where d
"""
function denormalize_dist_along(
    dist_normalized::Float64, 
    nzer::Normalizer{d}, 
    dims::Int
)::Float64 where d
    return dist_normalized * nzer.gap[dims]
end # denormalize_dist_along()
