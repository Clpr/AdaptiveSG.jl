export Node, NodeValue
export AdaptiveSparseGrid
export RegularSparseGrid
export YellowPages
export LinearStencil
export Normalizer



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
- `selfcontained::Bool`: if `true`, then the grid is self-contained, i.e. every
node, along every dimension, can find its left and right neighbors. For boundary
nodes, their neighbors exist along all dimensions except the boundary ones. This
self-containess property is essential to construct difference operators.

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
"""
mutable struct AdaptiveSparseGrid{d}
    nv           ::Dictionary{Node{d}, NodeValue{d}}
    depth        ::Int
    max_depth    ::Int
    max_levels   ::NTuple{d, Int}
    rtol         ::Float64
    selfcontained::Bool
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
mutable struct RegularSparseGrid{d}
    nv           ::Dictionary{Node{d}, NodeValue{d}}
    max_levels   ::NTuple{d, Int}
    max_depth    ::Int
end


# ------------------------------------------------------------------------------
"""
    YellowPages{d}

A `d`-dim yellow pages structure to store the by-dim left and right neighbors of
each grid point in an adaptive sparse grid. The "address", i.e. the integer inde
xes of each grid is also stored.

## Fields
- `address::Dictionary{Node{d}, Int}`: the address of each grid point in the ada
ptive sparse grid which is consistent to the matrix generated by `matrixize_*()`
methods
- `left::Matrix{Node{d}}`: the left neighbors of each grid point in the adaptive
sparse grid. It is a matrix of size `N x d`, where `N` is the number of grid po-
ints in the adaptive sparse grid
- `right::Matrix{Node{d}}`: the right neighbors of each grid point in the adapt-
ive sparse grid, also a matrix of size `N x d`
- `type::Symbol`: the type of the yellow pages, either `:ghost` or `:sparse`

## Notes
- Such yellow pages can be used to save ghost neighbor nodes. It can also save
the sparse neighbor nodes.
- If a node has no valid neighbor along some dimensions, then the corresponding
entry in the `left` or `right` is an invalid default node.
- To check if a `YellowPages{d}` matches a specific `AdaptiveSparseGrid{d}`, one
can compare the keys of `address` and the keys of the `nv` of the ASG. This is
a necessary condition, but not sufficient.
- The yellow pages do NOT save the distance to neighbor points. One has to compu
te the distance themselves.
- We use `Matrix` to save the neighbors for memory efficiency because a yellow
pages can only be defined after an ASG is trained.
- Depending on what type of neighbors you want to save, the yellow pages may
require self-containess of the grid if you want to save sparse neighbors. For
ghost neighbors, the self-containess is not necessary. This is checked by the
constructor of the yellow pages.
- This struct is element-wise mutable and passed by reference.
"""
struct YellowPages{d}
    address ::Dictionary{Node{d}, Int}
    left    ::Matrix{Node{d}}
    right   ::Matrix{Node{d}}
    type    ::Symbol
end # YellowPages{d}


# ------------------------------------------------------------------------------
"""
    LinearStencil{d}

A generic `m`-point stencil for a linear operator in `d` dimensions.

Let `Q` be a set of grid nodes; let `f:Q->R` be a real function; Let `L⊆Q` be a
subset of total `m` nodes. Then, consider linear operation: `∑_{q∈L} w(q) f(q)`,
in which `w(q)` are the weights. Such a linear operation can be represented by a
stencil of nodes and weights. With such a stencil, one can do this operation for
any function `f` defined on the grid.

## Fields
- `weights::Dictionary{Node{d}, Float64}`: a mapping from the nodes to the weig-
hts. The number of nodes in the stencil is simply the length of the dictionary.

## Notes
- The function `f` can also be a stacked vector of function values if someone
can locate the function values at the nodes.
- This struct is not a simple wrapper of a `Dictionary` because plus and minus
operations are defined on stencils (overloading `+` and `-` operators). The num-
ber of nodes in the new stencil must be in the range `[max(m1,m2),m1+m2]`
- The weights can be zero. But I would recommend to remove the node from the st-
encil unless it is necessary.
- The stencil struct cannot consistently ensure node alignment with another ste-
ncil that has the same number of nodes even though we use orderd dictionaries. 
If you do need precise alignment control, then be careful.
- This struct is useful when you handle e.g. finite difference operators on a
high dimensional Interpolant.
- The order of nodes in the stencil is supposed to be NOT important. One should
NOT rely on the order of nodes in the stencil.
- This struct is element-wise mutable and passed by reference.

## LinearStencil vs. SparseVector
- If all used nodes are in the set of grid nodes of an ASG, then the two repres-
entations are equivalent in computation. However, the `LinearStencil` allows you
to use any nodes which may not be in the grid. This is useful when you want to
write a human-readable stencil in the naive time iteration (explicit method) of
a PDE solver, or useful when you have very complicated stencil arithmetic before
applying the final stencil.
"""
struct LinearStencil{d}
    weights::Dictionary{Node{d}, Float64}

    """
        LinearStencil{d}(weights::Dictionary{Node{d}, Float64})

    Create a `d`-dim linear stencil with the given weights. This is the default
    constructor of the `LinearStencil{d}` type.
    """
    function LinearStencil{d}(weights::Dictionary{Node{d}, Float64}) where d
        new{d}(weights)
    end
end # LinearStencil{d}


# ------------------------------------------------------------------------------
"""
    Normalizer{d}

A generic `d`-dim normalizer type to save the min and max values of each dimens-
ion. It is used to normalize the input data to the domain `[0,1]^d` and reverse
the normalization.

## Fields
- `lb::NTuple{d, Float64}`: the minimum values of each dimension
- `gap::NTuple{d, Float64}`: maximum - minimum values of each dimension

## Notes
- Default constructor: `Normalizer{d}(lb, ub)`
- This struct is immutable and passed by values.
"""
struct Normalizer{d}
    lb ::NTuple{d, Float64}
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
        return new{d}(lb, ub .- lb)
    end
end # Normalizer{d}


















# ------------------------------------------------------------------------------
function Base.show(io::IO, n::Node{d}) where d
    print(io, "Node{", d, "}($(n.depth); $(n.ls); $(n.is))")
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, nv::NodeValue{d}) where d
    print(io, "NodeValue{", d, "}(nodal = $(nv.f), hierarchical = $(nv.α))")
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, asg::AdaptiveSparseGrid{d}) where d
    print(
        io, 
        "AdaptiveSparseGrid{", d, "}(depth = ", asg.depth, 
        ", rtol = ", asg.rtol, 
        ", selfcontained = ", asg.selfcontained, ")"
    )
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, rsg::RegularSparseGrid{d}) where d
    print(
        io, 
        "RegularSparseGrid{", d, 
        "}(depth = ", rsg.max_depth, ", ",
        "max_levels = ", rsg.max_levels, ")"
    )
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, yp::YellowPages{d}) where d
    print(
        io, 
        "YellowPages{", d, "}(#nodes = ", length(yp.address), 
        ", type = ", yp.type, ")"
    )
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, cls::LinearStencil{d}) where {d}
    print(io, "LinearStencil{", d, "}(#nodes = ", length(cls.weights), ")")
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, n::Normalizer{d}) where d
    print(io, "Normalizer{", d, "}(min = $(n.lb), max = $(n.lb .+ n.gap))")
end