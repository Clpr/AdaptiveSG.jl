export Node, NodeValue
export AbstractSparseGrid, AbstractLinearStencil
export AdaptiveSparseGrid
export RegularSparseGrid
export Normalizer


# ------------------------------------------------------------------------------
"""
    AbstractSparseGrid{d} <: Any

An abstract type for sparse grid types. It is used to define the common fields
and methods for sparse grid types.
"""
abstract type AbstractSparseGrid{d} <: Any end


# ------------------------------------------------------------------------------
"""
    AbstractLinearStencil{d} <: Any

An abstract type for linear stencil types.
"""
abstract type AbstractLinearStencil{d} <: Any end


# ------------------------------------------------------------------------------
"""
    Node{d}

A generic `d`-dim node type for either nodal or hierarchical nodes. There is no 
need to save the parent node neither the childrens, since they can be determined
by math operations given the hierarchy of the nodes.

## Fields
- `ls::NTuple{d, Int}`: level by dimension
- `is::NTuple{d, Int}`: index by dimension
- `depth::Int`: depth of the node in the hierarchy, `:=sum(ls)-d+1`

## Notes
- A node is uniquely identified by its level and index in a hyper cube [0,1]^d.
- This struct is immutable and passed by values.
"""
struct Node{d}
    ls   ::NTuple{d, Int}
    is   ::NTuple{d, Int}
    depth::Int
    function Node{d}(ls::NTuple{d, Int}, is::NTuple{d, Int}) where d
        new{d}(ls, is, nodedepth(ls))
    end
end # Node{d}


# ------------------------------------------------------------------------------
"""
    NodeValue{d}

A generic `d`-dim node value type. It is used to save the function value `f`, th
e hierarchical coefficient `αH`, and the grid point value `x` (in [0,1]^d)

## Fields
- `f::Float64`: function value at the grid point (= nodal coefficient value)
- `α::Float64`: hierarchical coefficient value

## Notes
- This struct is immutable and passed by values.
"""
struct NodeValue{d}
    f::Float64
    α::Float64
end # NodeValue{d}


# ------------------------------------------------------------------------------
"""
    Normalizer{d}

A generic `d`-dim normalizer type to save the min and max values of each dimens-
ion. It is used to normalize the input data to the domain `[0,1]^d` and reverse
the normalization.

## Fields
- `lb::NTuple{d, Float64}`: the minimum values of each dimension
- `ub::NTuple{d, Float64}`: the maximum values of each dimension
- `gap::NTuple{d, Float64}`: maximum - minimum values of each dimension

## Notes
- Default constructor: `Normalizer{d}(lb, ub)`
- This struct is immutable and passed by values.
"""
struct Normalizer{d}
    lb ::NTuple{d, Float64}
    ub ::NTuple{d, Float64}
    gap::NTuple{d, Float64}
    
    """
        Normalizer{d}(lb::NTuple{d, Float64}, ub::NTuple{d, Float64})

    Create a `d`-dim normalizer with the given min and max values of each dimen-
    sion. The min values must be less than the max values. Otherwise, an error
    will be thrown.
    """
    function Normalizer{d}(
        lb::NTuple{d, Float64}, 
        ub::NTuple{d, Float64}
    ) where d
        if any(lb .>= ub); throw(ArgumentError("min >= max found")); end
        return new{d}(lb, ub, ub .- lb)
    end
end # Normalizer{d}


# ------------------------------------------------------------------------------
"""
    AdaptiveSparseGrid{d}

A generic `d`-dim adaptive sparse grid type. It, conceptually modeled as a 2d-tr
ee, is saved as an ordered dictionary of hierarchical coefficients, where the ke
ys are the `Node{d}`.

## Fields
- `nv::Dictionary{Node{d}, NodeValue{d}}`: a mapping from `Node{d}` to the inte-
rpolant nodal and hierarchical coefficients
- `depth::Int`: the depth of the grid, `:=maximum(get_depth.(keys(nv)))`
- `max_depth::Int`: the maximum depth of the grid possible. The tree is not allo
wed to grow beyond this depth
- `max_levels::NTuple{d, Int}`: the maximum levels along each dimension
- `rtol::Float64`: the relative tolerance for the hierarchical coefficients, wh-
ich is the threshold for adding a new node. If the training process converges,
then for all points `x` in the domain, the relative interpolation error is no
greater than `rtol`. e.g. `rtol=1e-2` means that the error is less than 1%.
- `atol::Float64`: the absolute tolerance for the hierarchical coefficients, wh-
ich is the threshold for adding a new node. 

## Notes
- f(x): [0,1]^d -> R
- Use ordered hash map by `Dictionaries.jl` for evaluation performance
- the re-sacling of the grid to [0,1]^d is done by the user always
- boundaries are assumed to be non-zero (non-diminishing) always
- `depth` >= 2 required for any trained ASG. If not trained, then `depth=0` set
by the constructor.
- `max_depth` >=2 also required. The grid is not allowed to grow beyond it.
- For multiple dimension functions, the order of nodes in `nv` is not guaranteed
to be sorted by dimension. This tip reminds of using scatter plots.
- This struct is mutable and passed by reference.
- The `max_levels` provides information about the underlying equivalent dense
grid, or grid refinement, along each dimension. It determines the ghost mesh
step size in the ghost node method.
- Either `rtol` or `atol` is NaN if not used. If both are NaN, then the grid is
invalid.
"""
mutable struct AdaptiveSparseGrid{d} <: AbstractSparseGrid{d}
    nv           ::Dictionary{Node{d}, NodeValue{d}}
    depth        ::Int
    max_depth    ::Int
    max_levels   ::NTuple{d, Int}
    rtol         ::Float64
    atol         ::Float64
    use_rtol     ::Bool
end # AdaptiveSparseGrid{d}


# ------------------------------------------------------------------------------
"""
    RegularSparseGrid{d}

A generic `d`-dim regular sparse grid (RSG) type. It is a special case of the
ASG where the grid is regular.

## Fields
- `nv::Dictionary{Node{d}, NodeValue{d}}`: a mapping from `Node{d}` to the inte-
rpolant nodal and hierarchical coefficients
- `max_levels::NTuple{d, Int}`: the maximum levels along each dimension
- `max_depth::Int`: the max depth of the grid, i.e. accurary/refinement level

## Notes
- An RSG can be adapted to an ASG using `adapt!()` method; or simply convert it
to an ASG using `AdaptiveSparseGrid{d}(rsg)`.
- One can show that the depth of the tree representation is equal to the so call
"accuracy/refinement level" in the literature.
"""
mutable struct RegularSparseGrid{d} <: AbstractSparseGrid{d}
    nv           ::Dictionary{Node{d}, NodeValue{d}}
    max_levels   ::NTuple{d, Int}
    max_depth    ::Int
end













# ------------------------------------------------------------------------------
function Base.show(io::IO, n::Node{d}) where d
    print(io, "Node{", d, "}($(n.depth); $(n.ls); $(n.is))")
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, nv::NodeValue{d}) where d
    print(io, "NodeValue{", d, "}(nodal = $(nv.f), hierarchical = $(nv.α))")
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, n::Normalizer{d}) where d
    println(io, "Normalizer{", d, "}")
    for i in 1:d
        _lb = round(n.lb[i], digits=4)
        _ub = round(n.ub[i], digits=4)
        println(io, "\tx[$i] in [$_lb, $_ub]")
    end
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, asg::AdaptiveSparseGrid{d}) where d
    print(
        io, 
        "AdaptiveSparseGrid{", d, "}(depth = ", asg.depth, 
        ", #nodes = ", length(asg.nv),
        ", rtol = ", asg.rtol, 
        ", atol = ", asg.atol, 
        ", use_rtol = ", asg.use_rtol, ")"
    )
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, rsg::RegularSparseGrid{d}) where d
    print(
        io, 
        "RegularSparseGrid{", d, 
        "}(depth = ", rsg.max_depth, ", ",
        "#nodes = ", length(rsg.nv), ", ",
        "max_levels = ", rsg.max_levels, ")"
    )
end
