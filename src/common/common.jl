export isbig2add
export validate_f2fit!


# ------------------------------------------------------------------------------
"""
    isbig2add(α::Float64, fx::Float64, rtol::Float64)::Bool

Check if the given coefficient `α` is big enough to add a new node, using relat-
ive tolerance `rtol`.

## Notes
- the check is scaled by the function value at the candidate child node
- multi-linear interpolation implies phi = 1 at the grid point, so the nodal coe
fficient and the hierarchical coefficient is comparable in magnitude, which can
be conceptually seen as 1 - R^2
- we specically deal with the case `fx == 0` by using absolute tolerance instead
- If your function is zero in many places while the tolerance is large (e.g. 1%)
then this rule will lead to more unnecessary nodes at these places. This may be
improved in the future.
"""
function isbig2add(α::Float64, fx::Float64, rtol::Float64)::Bool
    if fx == 0
        return abs(α) > rtol
    else
        return (abs(α) / abs(fx)) > rtol
    end
end # isbig2add()


# ------------------------------------------------------------------------------
"""
    validate_f2fit!(f2fit::Function, d::Int)::Nothing

Check if the given function `f2fit` is valid for the training process and other
operations. It throws errors if the function does not meet the requirements.

Argument `d` is the dimension of the ASG that `f2fit` is supposed to be fitted.

## Rules
- Input: must have a method/implementation that be able to receive only one pos-
ition argument of type `SVector{d, Float64}`. This can be ensured if `f2fit` is
designed for generic vector-like types, or `AbstractVector{Float64}`.
- Output: must return a single `Float64` value.

## Notes
- Inline anonymous functions are supported.
"""
function validate_f2fit!(f2fit::Function, d::Int)::Nothing
    # input
    if !hasmethod(f2fit, Tuple{SVector{d,Float64}})
        throw(ArgumentError(string(
            "f2fit must have a method that is able to receive only ",
            "one single SVector{d,Float64} position argument."
        )))
    end
    # output
    _rettype = Base.return_types(f2fit, Tuple{SVector{d,Float64}})
    if (length(_rettype) != 1) | (_rettype[1] != Float64)
        throw(ArgumentError(string(
            "f2fit must return a single Float64 value but got $(_rettype)"
        )))
    end
    return nothing
end # validate_f2fit!()








