# A convenient wrapper (will replace the current API)


const Iterable = Union{AbstractVector, Tuple}


# ------------------------------------------------------------------------------
"""
    ScaledRSG{D}

## Example

```julia
asg = include("src/AdaptiveSG.jl")

m = asg.ScaledRSG{5}(5, lb = fill(-1.0, 5), ub = fill(1.0, 5))

asg.train!(m, x -> sum(x))

m(rand(5))

m.(rand(100,5) |> eachrow)

asg.basis_matrix(m, rand(5))

asg.interpcoef(m)

asg.update_all!(m, m |> length |> rand, printlevel = "final")

size(m)

length(m)

ndims(m)
```

"""
mutable struct ScaledRSG{D}
    domain::Normalizer{D}
    sg::RegularSparseGrid{D}

    function ScaledRSG{D}(
        accuracy ::Int ;
        maxlevels::NTuple{D,Int} = ntuple(_ -> accuracy, D),
        lb       ::Iterable = zeros(D),
        ub       ::Iterable = ones(D),
    ) where D
        @assert length(lb) == D "lb must have length $D"
        @assert length(ub) == D "ub must have length $D"
        new{D}(
            Normalizer{D}(lb |> NTuple{D,Float64}, ub |> NTuple{D,Float64}),
            RegularSparseGrid{D}(accuracy, maxlevels),
        )
    end
end
# ------------------------------------------------------------------------------
function (srsg::ScaledRSG{D})(x::AbstractVector ; extrapolation::Bool = true) where D
    return evaluate(srsg.sg, normalize(x, srsg.domain), extrapolation = extrapolation)
end
# ------------------------------------------------------------------------------
function train!(
    srsg::ScaledRSG{D},
    f2fit::Function ;
    printlevel::String = "final",
) where D
    train!(
        srsg.sg,
        x -> f2fit(denormalize(x, srsg.domain)),
        printlevel = printlevel,
        validate_io = false,
    )
    return nothing
end
# ------------------------------------------------------------------------------
function Base.ndims(srsg::ScaledRSG{D}) where D
    return D
end
# ------------------------------------------------------------------------------
function Base.length(srsg::ScaledRSG{D}) where D
    return length(srsg.sg)
end
# ------------------------------------------------------------------------------
function Base.size(srsg::ScaledRSG{D}) where D
    return (length(srsg),D)
end
# ------------------------------------------------------------------------------
function Base.stack(srsg::ScaledRSG{D}) where D
    X01 = vectorize_x(srsg.sg)
    for x01i in srsg.sg |> eachrow
        x01i .= denormalize(x01i, srsg.domain)
    end
    return X01
end
# ------------------------------------------------------------------------------
function update_all!(
    srsg::ScaledRSG{D}, 
    Ynew::AbstractVector ;
    printlevel::String = "final",
) where D
    @assert length(srsg.sg) == length(Ynew) "Ynew must have length $(length(srsg.sg))"

    update_all!(
        srsg.sg,
        Dictionary{Node{D}, Float64}(
            srsg.sg.nv |> keys,
            Ynew,
        ),
        printlevel = printlevel,
    )

    return nothing
end
# ------------------------------------------------------------------------------
function basis_matrix(srsg::ScaledRSG{D}, x::AbstractVector) where D
    return basis_matrix(
        srsg.sg,
        normalize(x, srsg.domain),
    )
end
# ------------------------------------------------------------------------------
function interpcoef(srsg::ScaledRSG{D}) where D
    return interpcoef(srsg.sg)
end





