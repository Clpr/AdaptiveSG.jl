export normalize, denormalize
export normalize_dist_along, denormalize_dist_along


# ------------------------------------------------------------------------------
"""
    normalize(
        x::AbstractVector{Float64}, 
        nzer::Normalizer{d}
    )::SVector{d,Float64} where d

Normalize the input vector `x` to the domain `[0,1]^d` using the `Normalizer{d}`
"""
function normalize(
    x::AbstractVector, 
    nzer::Normalizer{d}
)::SVector{d,Float64} where d
    if length(x) != d; throw(ArgumentError("length(x) = $(length(x)) != d = $d")); end
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
function denormalize(node::Node{d},nzer::Normalizer{d}) where d
    return denormalize(node |> get_x, nzer)
end


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
function denormalize(node::Node{d},nzer::Normalizer{d},dims::Int) where d
    return denormalize(node |> get_x, nzer, dims)
end


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






# ------------------------------------------------------------------------------
"""
    normalize(f2fit::Function, nzer::Normalizer{d})::Function where d

Decorates a function `f2fit` that takes a single vector argument of type
`AbstractVector{Float64}` in a `d`-dim hyper-rectangle space to a function that
takes a single vector argument of type `SVector{d,Float64}` in the hypercube
`[0,1]^d` space.

## Arguments
- `f2fit::Function`: The function to be decorated.
- `nzer::Normalizer{d}`: The normalizer object that defines the original domain
as a hyper-rectangle.

## Returns
- A decorated function that takes a single vector argument of type 
`SVector{d,Float64}` in the hypercube `[0,1]^d` space, and the output is forced
to be a `Float64` type.
"""
function normalize(f2fit::Function, nzer::Normalizer{d})::Function where d
    # function _decorated(X01::SVector{d,Float64})::Float64
    #     return f2fit(denormalize(X01, nzer))
    # end
    return X01::SVector{d,Float64} -> Float64(f2fit(denormalize(X01, nzer)))
end







# ------------------------------------------------------------------------------
"""
    clamp(x::Real, nzer::Normalizer{d}, dims::Int) where d

Clamp the input scalar `x` to the domain `[0,1]` using the `Normalizer{d}`,
where `dims` is the dimension of the scalar belonging to.
"""
function Base.clamp(x::Real, nzer::Normalizer{d}, dims::Int) where d
    return clamp(x, nzer.lb[dims], nzer.ub[dims])
end
# ------------------------------------------------------------------------------
function Base.clamp(
    x   ::AbstractVector,
    nzer::Normalizer{d}
) where d
    return clamp.(x, nzer.lb, nzer.ub)
end





# ------------------------------------------------------------------------------
function Base.LinRange(nzer::Normalizer{d}, dims::Int, n::Int)::LinRange where d
    return LinRange(nzer.lb[dims], nzer.ub[dims], n)
end





# ------------------------------------------------------------------------------
function Base.rand(nzer::Normalizer{d}, n::Int)::Matrix{Float64} where {d}
    # RANDOM NUMBER IN A SPECIFIC HYPER-RECTANGLE DOMAIN
    # nzer: Nzer{D} - a domain ⊂ R^D
    # n: Int - number of random numbers

    X = rand(n, d)

    # scaling each column
    for j in 1:d
        X[:,j] = nzer.lb[j] .+ X[:,j] .* (nzer.ub[j] - nzer.lb[j])
    end # j

    return X
end # rand