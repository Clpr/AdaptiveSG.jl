export get_neighbor


# ------------------------------------------------------------------------------
function get_neighbor(
    node::Node{d}, 
    yp::YellowPages{d}, 
    dims::Int,
    direction::Symbol
)::Node{d} where d
    if direction == :left
        return yp.left[yp.address[node], dims]
    elseif direction == :right
        return yp.right[yp.address[node], dims]
    else
        throw(ArgumentError("invalid direction"))
    end
end # get_neighbor()




