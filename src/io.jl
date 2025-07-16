#===============================================================================
I/O API
===============================================================================#
export serialize
export deserialize


# ------------------------------------------------------------------------------
SerializedSG = Dict{
    Symbol,
    Union{
        Int,
        Float64,
        Vector{Float64},
        Matrix{Int},
        Vector{Int},
    }
}
# ------------------------------------------------------------------------------
"""
    serialize(G::SparseGridInterpolant{D}) where {D}

Serialize the sparse grid interpolant `G` into a dictionary.
"""
function serialize(G::SparseGridInterpolant{D}) where {D}
    dat = SerializedSG()

    dat[:D] = D

    dat[:lb] = G.lb |> collect
    dat[:ub] = G.ub |> collect

    dat[:levels]  = G.levels
    dat[:indices] = G.indices
    dat[:depths]  = G.depths

    dat[:fvals] = G.fvals
    dat[:coefs] = G.coefs

    return dat
end
# ------------------------------------------------------------------------------
"""
    deserialize(dat::SerializedSG)::SparseGridInterpolant

Deserialize the serialized sparse grid interpolant `dat` into a 
`SparseGridInterpolant` object.
"""
function deserialize(dat::SerializedSG)::SparseGridInterpolant
    @assert haskey(dat, :lb) "Serialized data must contain lower bound."
    @assert haskey(dat, :ub) "Serialized data must contain upper bound."
    @assert haskey(dat, :levels) "Serialized data must contain max_levels."
    @assert haskey(dat, :indices) "Serialized data must contain indices."
    @assert haskey(dat, :depths) "Serialized data must contain depths."
    @assert haskey(dat, :fvals) "Serialized data must contain fvals."
    @assert haskey(dat, :coefs) "Serialized data must contain coefs."

    G = SparseGridInterpolant(
        dat[:D],
        lb = dat[:lb],
        ub = dat[:ub],
    )

    G.levels = dat[:levels]
    G.indices = dat[:indices]
    G.depths = dat[:depths]
    
    G.fvals = dat[:fvals]
    G.coefs = dat[:coefs]

    return G
end




