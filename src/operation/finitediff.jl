export get_spvec_node

export get_stencil_node
# export get_stencil_gradient
# export get_stencil_hessian
export get_dehierarchization_matrix
export get_hierarchization_matrix


# ------------------------------------------------------------------------------
"""
    get_spvec_node(
        p::Node{d},
        G::AdaptiveSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64  = 1e-16
    )::SparseVector{Float64,Int}

Create a `SparseVector{Float64,Int}` of representing a node using all the nodes
in an ASG `G`. The `p` node can be any node but not necessarily in the grid `G`.

## Arguments
- `p::Node{d}`: The node to be represented.
- `G::AdaptiveSparseGrid{d}`: The ASG.
- `dropzeros::Bool`: Drop the zero weights. Default is `true`.
- `atol::Float64`: Abs tolerance to consider a weight as zero. Default `1e-16`.
"""
function get_spvec_node(
    p::Node{d},
    G::AdaptiveSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64  = 1e-16
)::SparseVector{Float64,Int} where {d}
    Is = Int[]
    Vs = Float64[]
    x0::SVector{d,Float64} = get_x(p)
    for (i, p2) in enumerate(keys(G.nv))
        w = ϕ_unsafe(x0, p2)
        if dropzeros && isapprox(w, 0.0, atol = atol); continue; end
        push!(Is, i)
        push!(Vs, w)
    end
    return sparse(Is, Vs, length(G))
end # get_spvec_node()


# ------------------------------------------------------------------------------
"""
    get_stencil_node(
        node::Node{d},
        G::AdaptiveSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64  = 1e-16
    )::LinearStencil{d} where d

Create a `LinearStencil{d}` of the interpolant at the given node `node`. The st-
encil weights are `ϕ(x, node)`. And the stencil should directly apply to the hi-
erarchical coefficients in the ASG using `apply(stc,G,applyto=:hierarchical)`.
Applying this stencil leads to the same result as `evaluate(G, get_x(node))`. 

## Arguments
- `node::Node{d}`: The node at which the interpolant is to be calculated.
- `G::AdaptiveSparseGrid{d}`: The ASG.
- `dropzeros::Bool`: Drop the zero weights. Default is `true`.
- `atol::Float64`: Abs tolerance to consider a weight as zero. Default `1e-16`.

## Notes
- To force the stencil to include all nodes, set `dropzeros = false`, or set the
`atol` to a very small value such as `Inf`
- This function constructs the stencil of the de-hierarchization operator at the
given `node`. The de-hierarchization operator corresponds to the notation `E` in
the sparse grid literature.
"""
function get_stencil_node(
    node::Node{d},
    G::AdaptiveSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64   = 1e-16
)::LinearStencil{d} where d
    x0        = get_x(node)
    usednodes = Node{d}[]
    weights   = Float64[]
    for p in keys(G.nv)
        _w = ϕ_unsafe(x0, p)
        if dropzeros && isapprox(_w, 0.0, atol = atol)
            continue
        else
            push!(usednodes, p)
            push!(weights, _w)
        end
    end
    return LinearStencil{d}(usednodes, weights)
end # get_stencil_node()


# # ------------------------------------------------------------------------------
# """
#     get_stencil_gradient(
#         G        ::AdaptiveSparseGrid{d},
#         yp       ::YellowPages{d},
#         node     ::Node{d},
#         dims     ::Int ;
#         direction::Symbol        = :forward,
#         nzer     ::Normalizer{d} = Normalizer{d}()
#     )::LinearStencil{d} where d

# Create a `LinearStencil{d}` of the gradient of the interpolant at the given node
# `node`. 

# ## Arguments
# - `G::AdaptiveSparseGrid{d}`: The ASG.
# - `yp::YellowPages{d}`: The yellow pages constructed from `G`.
# - `node::Node{d}`: The node at which the gradient is to be calculated.
# - `dims::Int`: The dimension along which the gradient is to be calculated.
# - `direction::Symbol`: The direction of the gradient. Default is `:forward`, can
# be `:backward` or `:central` as well
# - `nzer::Normalizer{d}`: The normalizer to be used. Default is `Normalizer{d}()`
# which is the identity normalizer (hypercube `[0,1]^d` domain).


# ## Notes
# - This method is designed for getting the gradient at exact node in the grid. If
# you need the gradient at an arbitrary point in the domain, write your own method
# by calling `evaluate()` multiple times.
# - Only the typical two-point forward/backward/central difference is implemented,
# where forward/backward are `O(Δx)` and central is `O(Δx²)`. Higher order approx-
# imations may be implemented in the future (as a keyword argument).
# - Requires the `YellowPages{d}` to avoid repeated calculations.
# - The stencil is adjusted for uneven grid spacing when `direction == :central`
# - To apply the stencil to the ASG, use `apply()` later.
# """
# function get_stencil_gradient(
#     G        ::AdaptiveSparseGrid{d},
#     yp       ::YellowPages{d},
#     node     ::Node{d},
#     dims     ::Int ;
#     direction::Symbol        = :forward,
#     nzer     ::Normalizer{d} = Normalizer{d}()
# )::LinearStencil{d} where d
#     if !haskey(G.nv, node); throw(ArgumentError("node is not in the grid")); end
#     flgbnd  = get_boundaryflag(node, dims)
#     weights = Dictionary{Node{d}, Float64}()
#     if direction == :forward
#         if flgbnd == 1
#             throw(ArgumentError("right boundary node but direction is forward"))
#         end

#         # get right neighbor node
#         node_r = get_neighbor(node, yp, dims, :right)

#         # get difference step size, considering de-normalization
#         Δx      = get_dist_along(node, node_r, dims)
#         Δorigin = Δx / nzer.gap[dims]

#         # add: the right neighbor node with weight 1/Δorigin
#         insert!(weights, node_r, 1.0 / Δorigin)

#         # add: the current node with weight -1/Δorigin
#         insert!(weights, node, -1.0 / Δorigin)

#     elseif direction == :backward
#         if flgbnd == -1
#             throw(ArgumentError("left boundary node but direction is backward"))
#         end

#         node_l  = get_neighbor(node, yp, dims, :left)
#         Δx      = get_dist_along(node_l, node, dims)
#         Δorigin = Δx / nzer.gap[dims]

#         insert!(weights, node_l, -1.0 / Δorigin)
#         insert!(weights, node  , 1.0 / Δorigin)

#     elseif direction == :central
#         if flgbnd != 0
#             throw(ArgumentError("boundary node but direction is central"))
#         end

#         node_l = get_neighbor(node, yp, dims, :left)
#         node_r = get_neighbor(node, yp, dims, :right)
#         Δxl    = get_dist_along(node_l, node, dims)
#         Δxr    = get_dist_along(node, node_r, dims)
#         Δol    = Δxl / nzer.gap[dims]
#         Δor    = Δxr / nzer.gap[dims]

#         insert!(weights, node_l, -1.0 / (Δol + Δor))
#         insert!(weights, node_r,  1.0 / (Δol + Δor))

#     else
#         throw(ArgumentError("invalid direction"))
#     end
#     return LinearStencil{d}(weights)
# end # get_stencil_gradient()


# # ------------------------------------------------------------------------------
# """
#     get_stencil_hessian(
#         G        ::AdaptiveSparseGrid{d},
#         yp       ::YellowPages{d},
#         pc       ::Node{d},
#         dims     ::NTuple{2,Int} ;
#         direction::NTuple{2,Symbol} = (:forward, :forward),
#         nzer     ::Normalizer{d} = Normalizer{d}()
#     )::LinearStencil{d} where d

# Create a `LinearStencil{d}` of the Hessian of the interpolant at the given node
# `pc`. The Hessian is a matrix of second order partial derivatives.

# ## Arguments
# - `G::AdaptiveSparseGrid{d}`: The ASG.
# - `yp::YellowPages{d}`: The yellow pages constructed from `G`.
# - `pc::Node{d}`: The node at which the Hessian is to be calculated.
# - `dims::NTuple{2,Int}`: The dimensions along which the Hessian is to be calcul-
# ated. The first and the second elements are the dimensions `i` and `j` respecti-
# vely. The order of the dimensions matters, i.e. `dims[1] != dims[2]` in general.
# - `direction::NTuple{2,Symbol}`: The directions of the Hessian to be taken. The
# first and the second elements are the directions of the first and the second or-
# der derivatives respectively. Default is `(:forward, :forward)`, can be `:forwa-
# rd`, `:backward` or `:central` for each element. But, if `dims[1] == dims[2]`,
# then the directions must be the same.
# - `nzer::Normalizer{d}`: The normalizer to be used. Default is `Normalizer{d}()`
# which is the identity normalizer (hypercube `[0,1]^d` domain).
# """
# function get_stencil_hessian(
#     G        ::AdaptiveSparseGrid{d},
#     yp       ::YellowPages{d},
#     pc       ::Node{d},
#     dims     ::NTuple{2,Int} ;
#     direction::NTuple{2,Symbol} = (:forward, :forward),
#     nzer     ::Normalizer{d} = Normalizer{d}()
# )::LinearStencil{d} where d
#     if dims[1] == dims[2]
#         if direction[1] == direction[2]
#             return get_stencil_hessian_diagonal(
#                 G, 
#                 yp, 
#                 pc, 
#                 dims[1], 
#                 direction = direction[1], 
#                 nzer = nzer
#             )
#         else
#             throw(ArgumentError("Hessian diagonal:directions must be the same"))
#         end
#     else
#         return get_stencil_hessian_offdiagonal(
#             G, 
#             yp, 
#             pc, 
#             dims, 
#             direction = direction, 
#             nzer = nzer
#         )
#     end
# end # get_stencil_hessian()




# # ------------------------------------------------------------------------------
# """
#     get_stencil_hessian_diagonal(
#         G        ::AdaptiveSparseGrid{d},
#         yp       ::YellowPages{d},
#         pc       ::Node{d},
#         dims     ::Int ;
#         direction::Symbol        = :forward,
#         nzer     ::Normalizer{d} = Normalizer{d}()
#     )::LinearStencil{d} where d

# Create a `LinearStencil{d}` of the diagonal of the Hessian of the interpolant at
# the given node `pc`.
# """
# function get_stencil_hessian_diagonal(
#     G        ::AdaptiveSparseGrid{d},
#     yp       ::YellowPages{d},
#     pc       ::Node{d},
#     dims     ::Int ;
#     direction::Symbol        = :forward,
#     nzer     ::Normalizer{d} = Normalizer{d}()
# )::LinearStencil{d} where d
#     if !haskey(G.nv, pc); throw(ArgumentError("node is not in the grid")); end
#     flgbnd  = get_boundaryflag(pc, dims)
#     weights = Dictionary{Node{d}, Float64}()
#     if direction == :forward
#         pr1 = get_neighbor(pc , yp, dims, :right)
#         pr2 = get_neighbor(pr1, yp, dims, :right)
#         if !isvalid(pr2)
#             # NOTES: if pr1 is already invalid, then `get_neighbor(pr1)` will
#             # throw an error. So we don't need to check pr1 here.
#             throw(ArgumentError("2nd right neighbor does not exist"))
#         end
#         Δor1 = get_dist_along(pc, pr1, dims) / nzer.gap[dims]
#         Δor2 = get_dist_along(pr1, pr2, dims) / nzer.gap[dims]

#         # add: the current (center) node
#         insert!(weights, pc, 2.0 / (Δor1 * (Δor1 + Δor2)) )
#         # add: the right neighbor node
#         insert!(weights, pr1, -2.0 / (Δor1 * Δor2))
#         # add: the 2nd right neighbor node
#         insert!(weights, pr2, 2.0 / (Δor2 * (Δor1 + Δor2)))

#     elseif direction == :backward
#         if flgbnd == -1
#             throw(ArgumentError("left boundary node but direction is backward"))
#         end
#         pl1 = get_neighbor(pc , yp, dims, :left)
#         pl2 = get_neighbor(pl1, yp, dims, :left)
#         if !isvalid(pl2)
#             throw(ArgumentError("2nd left neighbor does not exist"))
#         end
#         Δol1 = get_dist_along(pl1, pc, dims)  / nzer.gap[dims]
#         Δol2 = get_dist_along(pl2, pl1, dims) / nzer.gap[dims]

#         # add: the current (center) node
#         insert!(weights, pc, 2.0 / (Δol1 * (Δol1 + Δol2)) )
#         # add: the left neighbor node
#         insert!(weights, pl1, -2.0 / (Δol1 * Δol2))
#         # add: the 2nd left neighbor node
#         insert!(weights, pl2, 2.0 / (Δol2 * (Δol1 + Δol2)))

#     elseif direction == :central
#         if (flgbnd != 0)
#             throw(ArgumentError("boundary node but direction is central"))
#         end
#         pl1  = get_neighbor(pc, yp, dims, :left)
#         pr1  = get_neighbor(pc, yp, dims, :right)
#         Δol1 = get_dist_along(pl1, pc, dims) / nzer.gap[dims]
#         Δor1 = get_dist_along(pc, pr1, dims) / nzer.gap[dims]

#         # add: the current (center) node
#         insert!(weights, pc, -2.0 / (Δol1 * Δor1))
#         # add: the left neighbor node
#         insert!(weights, pl1, 2.0 / (Δol1 * (Δol1 + Δor1)))
#         # add: the right neighbor node
#         insert!(weights, pr1, 2.0 / (Δor1 * (Δol1 + Δor1)))

#     else
#         throw(ArgumentError("invalid direction"))
#     end
#     return LinearStencil{d}(weights)
# end # get_stencil_hessian_diagonal()



# # ------------------------------------------------------------------------------
# # for off-diagonal second order derivatives (Hessian) ∂2f/∂xi∂xj
# function get_stencil_hessian_offdiagonal(
#     G        ::AdaptiveSparseGrid{d},
#     yp       ::YellowPages{d},
#     node     ::Node{d},
#     dims     ::NTuple{2,Int} ; # (x,y) in order; ∂2f/∂x∂y != ∂2f/∂y∂x in general
#     direction::NTuple{2,Symbol} = (:forward, :forward),
#     nzer     ::Normalizer{d}    = Normalizer{d}()
# )::LinearStencil{d} where d
#     if !haskey(G.nv, node); throw(ArgumentError("node is not in the grid")); end
#     flgbnd = (
#         get_boundaryflag(node, dims[1]),
#         get_boundaryflag(node, dims[2])
#     )

#     # check the node's position on the (x,y) plane; then determine which directs
#     # are feasible.

#     # 2D plane illustration:
#     #
#     # (y increases)
#     # ^
#     # |
#     # | (topleft) ===== (top) ===== (topright)
#     # |     ||           ||            ||
#     # | (left) ===== (center) ===== (right)
#     # |     ||           ||            ||
#     # | (bottomleft) == (bottom) == (bottomright)
#     # | ------------------------------------------> (x increases)
#     #

#     # all possible combinations & feasible directions:
#     # (The mostly-used directions are marked with √)
#     #
#     #  - (-1,-1): bottomleft
#     #    - (:forward,  :forward)  √
#     # 
#     #  - (-1, 0): left
#     #    - (:forward,  :forward)
#     #    - (:forward,  :central)  √
#     #    - (:forward, :backward)
#     #
#     #  - (-1, 1): topleft
#     #    - (:forward,  :backward)  √
#     #
#     #  - ( 0,-1): bottom
#     #    - (:central,   :forward)  √
#     #    - (:forward,   :forward)
#     #    - (:backward,  :forward)
#     # 
#     #  - ( 0, 0): center (or interior)
#     #    - (:central,   :central)  √
#     #    - (:forward,   :central)
#     #    - (:backward,  :central)
#     #    - (:central,   :forward)
#     #    - (:forward,   :forward)
#     #    - (:backward,  :forward)
#     #    - (:central,  :backward)
#     #    - (:forward,  :backward)
#     #    - (:backward, :backward)
#     #
#     #  - ( 0, 1): top
#     #    - (:central,  :backward)  √
#     #    - (:forward,  :backward)
#     #    - (:backward, :backward)
#     #
#     #  - ( 1,-1): bottomright
#     #    - (:backward,  :forward)  √
#     # 
#     #  - ( 1, 0): right
#     #    - (:backward,  :forward)
#     #    - (:backward,  :central)  √
#     #    - (:backward, :backward)
#     #
#     #  - ( 1, 1): topright
#     #    - (:backward,  :backward)  √
#     #
#     # NOTES:
#     #  - Because the order of taking derivatives matters in an ASG, then the 2nd
#     #    direction more determines which nodes to use: It determines AT which
#     #    neighbor nodes are used to calculate the 1st-order derivatives at. The
#     #    1st-order derivatives are directly used to calculate the 2nd-order der-
#     #    ivatives. (Step 1 in algorithm)
#     #  - The 1st direction determines, conditional on knowing AT which nodes to
#     #    calculate the 1st-order derivatives, BY which nodes to calculate these
#     #    1st-order derivatives. (Step 2 in algorithm)
#     #    - In fact, more detailed control of Step 2 e.g. allowing different dir-
#     #      ections for different Step 1 nodes, is not implemented here. This de-
#     #      mand is rare in practice and hard to either understand or implement.
#     #  - We hard-code the feasible location-direction combinations above, and
#     #    throw an error if the user tries to use an invalid combination.
#     # 
#     # However, there is a simple rule to exclude the invalid combinations.

#     # Step 1: choose nodes AT which to calculate 1st-order derivatives along the
#     #         2nd dimension. These two nodes are thought "virtual"
#     # NOTES:  we know that offsets = either (0,1) or (-1,1) only.
#     # NOTES:  we name pv2 is righter than pv1 along the 2nd dimension
#     offsets = _private_hessian_offset_locate_points(flgbnd[2], direction[2])
#     if offsets == (0,1)
#         if direction[2] == :forward
#             pv1 = node
#             pv2 = get_neighbor(node, yp, dims[2], :right)
#         elseif direction[2] == :backward
#             pv1 = get_neighbor(node, yp, dims[2], :left)
#             pv2 = node
#         else
#             throw(ArgumentError("invalid direction"))
#         end
#     elseif offsets == (-1,1)
#         pv1 = get_neighbor(node, yp, dims[2], :left)
#         pv2 = get_neighbor(node, yp, dims[2], :right)
#     else
#         throw(ArgumentError("invalid offset"))
#     end


#     # Step 2: choose nodes BY which to calculate 1st-order derivatives along the
#     #         1st dimension. These four nodes are the real nodes.
#     # NOTES:  when either pv1 or pv2 is invalid (exceeding the boundary), then
#     #         this step will throw an error of no such node found in the yellow
#     #         pages. so no extra check is needed here.
#     # NOTES:  we name p12 is righter than p11, and p22 is righter than p21 along
#     #         the first dimension.
#     # NOTES:  a fact: `node` must have the same boundary flag along the 1st dim-
#     #         ension as `pv1` and `pv2` (in a self-contained ASG).
#     offsets = _private_hessian_offset_locate_points(flgbnd[1], direction[1])
#     if offsets == (0,1)
#         if direction[1] == :forward
#             p11 = pv1
#             p12 = get_neighbor(pv1, yp, dims[1], :right)
#             p21 = pv2
#             p22 = get_neighbor(pv2, yp, dims[1], :right)
#         elseif direction[1] == :backward
#             p11 = get_neighbor(pv1, yp, dims[1], :left)
#             p12 = pv1
#             p21 = get_neighbor(pv2, yp, dims[1], :left)
#             p22 = pv2
#         else
#             throw(ArgumentError("invalid direction"))
#         end
#     elseif offsets == (-1,1)
#         p11 = get_neighbor(pv1, yp, dims[1], :left)
#         p12 = get_neighbor(pv1, yp, dims[1], :right)
#         p21 = get_neighbor(pv2, yp, dims[1], :left)
#         p22 = get_neighbor(pv2, yp, dims[1], :right)
#     else
#         throw(ArgumentError("invalid offset"))
#     end


#     # Step 3: get the distances and do de-normalization
#     # distance between pv1 and pv2 along the 2nd dimension
#     Δv1_v2 = get_dist_along(pv1, pv2, dims[2]) / nzer.gap[dims[2]]
#     # distance between p11 and p12 along the 1st dimension
#     Δ11_12 = get_dist_along(p11, p12, dims[1]) / nzer.gap[dims[1]]
#     # distance between p21 and p22 along the 1st dimension
#     Δ21_22 = get_dist_along(p21, p22, dims[1]) / nzer.gap[dims[1]]

#     # Step 4: construct the stencil
#     # NOTES:  we use stencil arithmetic rather than directly constructing the
#     #         stencil for readability and avoiding explicitly merging overlappi-
#     #         ng nodes.
#     # ∂f/∂x(x,y-Δy): (f(p12) - f(p11)) / Δ11_12
#     stc1 = LinearStencil{d}(
#         Node{d}[p11, p12],
#         Float64[
#             -1.0 / Δ11_12,
#              1.0 / Δ11_12
#         ]
#     )
#     # ∂f/∂x(x,y+Δy): (f(p22) - f(p21)) / Δ21_22
#     stc2 = LinearStencil{d}(
#         Node{d}[p21, p22],
#         Float64[
#             -1.0 / Δ21_22,
#              1.0 / Δ21_22
#         ]
#     )
#     # ∂2f/∂x∂y(x,y): (∂f/∂x(x,y+Δy) - ∂f/∂x(x,y-Δy)) / Δv1_v2
#     stc3 = stc2 / Δv1_v2 - stc1 / Δv1_v2
    
#     return stc3
# end # get_stencil_hessian_offdiagonal()



# # ------------------------------------------------------------------------------
# """
#     _private_hessian_offset_locate_points(
#         flgbnd   ::Int,
#         direction::Symbol
#     )::Ntuple{2, Int}

# Private helper function for `get_stencil_hessian_offdiagonal()`. This function
# determines the two points' offset according to the boundary flag and the direct-
# ion. 

# `0` means the current node, `1` means the next node along the direction. `-1` 
# means the previous node along the direction.
# """
# function _private_hessian_offset_locate_points(
#     flgbnd   ::Int,
#     direction::Symbol
# )::NTuple{2, Int}
#     if flgbnd == -1
#         if direction == :forward
#             return (0,1)
#         else
#             throw(ArgumentError("invalid direction"))
#         end
#     elseif flgbnd == 0
#         return if direction == :forward
#             return (0,1)
#         elseif direction == :central
#             return (-1,1)
#         elseif direction == :backward
#             return (0,1) # same as forward, but opposite direction
#         else
#             throw(ArgumentError("invalid direction"))
#         end
#     elseif flgbnd == 1
#         if direction == :backward
#             return (0,1)
#         else
#             throw(ArgumentError("invalid direction"))
#         end
#     else
#         throw(ArgumentError("invalid boundary flag"))
#     end
# end # _private_hessian_offset_locate_points()


# ------------------------------------------------------------------------------
"""
    get_dehierarchization_matrix(
        G::AdaptiveSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64   = 1E-16
    )::SparseMatrixCSC{Float64} where d

Get the de-hierarchization matrix `E` of the ASG `G`. The de-hierarchization ma-
trix is a sparse matrix that can be applied to the hierarchical coefficients of
the ASG to get the nodal values (interpolated function values at nodes).

## Notes
- The `dropzeros` and `atol` arguments are used to avoid storing the zero values
in the matrix. The zero values are dropped if `dropzeros = true` and the values
are considered zero if `abs(value) < atol`.
- We drop zeros during the matrix construction rather than call `dropzeros` then
"""
function get_dehierarchization_matrix(
    G::AdaptiveSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64   = 1E-16
)::SparseMatrixCSC{Float64,Int} where d
    Is = Int[]; Js = Int[]; Vs = Float64[]
    n  = length(G)
    for (i, p) in enumerate(keys(G.nv))
        for (j, p2) in enumerate(keys(G.nv))
            w = ϕ_unsafe(get_x(p), p2)
            if dropzeros && isapprox(w, 0.0, atol = atol); continue; end
            push!(Is, i)
            push!(Js, j)
            push!(Vs, w)
        end # (j, p2)
    end # (i, p2)
    return sparse(Is, Js, Vs, n, n)
end # get_dehierarchization_matrix()


# ------------------------------------------------------------------------------
"""
    get_hierarchization_matrix(
        G::AdaptiveSparseGrid{d} ;
        dropzeros::Bool = true,
        atol::Float64   = 1E-16
    )::SparseMatrixCSC{Float64} where d

Get the hierarchization matrix `E` of the ASG `G`. The hierarchization matrix is
a sparse matrix that can be applied to the nodal coefficients.

However, we recommand to use "H * nodal_values = E \\ nodal_values" instead of
computing `H` explicitly (if possible).

## Notes
- IMPORTANT: The `lu()` for sparse matrix in Julia behaves differently from the
`lu()` for dense matrix, so we cannot directly use `E^{-1}=U^{-1}*L^{-1}`. There
is no solution yet.
"""
function get_hierarchization_matrix(
    G::AdaptiveSparseGrid{d} ;
    dropzeros::Bool = true,
    atol::Float64   = 1E-16
)::SparseMatrixCSC{Float64} where d
    E = get_dehierarchization_matrix(G, dropzeros = dropzeros, atol = atol)
    return sparse(E \ Matrix{Float64}(I, size(E)...))
end # get_hierarchization_matrix()
