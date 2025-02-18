# Serialization functions
# ------------------------------------------------------------------------------
export to_dict
export save
export parse
export load

# ------------------------------------------------------------------------------
"""
    to_dict(G::AdaptiveSparseGrid{d}) where {d}

Serialize the `AdaptiveSparseGrid` `G` to a dictionary.
"""
function to_dict(G::AdaptiveSparseGrid{d}) where {d}
    return Dict(
        "type" => "AdaptiveSparseGrid",
        "dim"  => d,
        "depth"        => G.depth,
        "max_depth"    => G.max_depth,
        "max_levels"   => G.max_levels,
        "atol"         => isnan(G.atol) ? -1.0 : G.atol,
        "rtol"         => isnan(G.rtol) ? -1.0 : G.rtol,
        "use_rtol"     => G.use_rtol,
        "levels"       => G |> vectorize_levels,
        "indices"      => G |> vectorize_indices,
        "nodal"        => G |> vectorize_nodal,
        "hierarchical" => G |> vectorize_hierarchical,
    )
end # to_dict
# ------------------------------------------------------------------------------
"""
    to_dict(G::RegularSparseGrid{d}) where {d}

Serialize the `AdaptiveSparseGrid` `G` to a dictionary.
"""
function to_dict(G::RegularSparseGrid{d}) where {d}
    return Dict(
        "type" => "RegularSparseGrid",
        "dim"  => d,
        "max_depth"   => G.max_depth,
        "max_levels"  => G.max_levels,
        "levels"       => G |> vectorize_levels,
        "indices"      => G |> vectorize_indices,
        "nodal"        => G |> vectorize_nodal,
        "hierarchical" => G |> vectorize_hierarchical,
    )
end # to_dict
# ------------------------------------------------------------------------------
"""
    to_dict(nzer::Normalizer{d}) where {d}

Serialize the `Normalizer` `nzer` to a dictionary.
"""
function to_dict(nzer::Normalizer{d}) where {d}
    return Dict(
        "type" => "Normalizer",
        "dim"  => d,
        "lb"   => nzer.lb,
        "ub"   => nzer.ub,
    )
end # to_dict







# ------------------------------------------------------------------------------
"""
    save(G::AbstractSparseGrid{d}, filename::String) where {d}

Save the `AbstractSparseGrid` `G` to a file `filename` in JSON format.
"""
function save(G::AbstractSparseGrid{d}, filename::String) where {d}
    open(filename, "w") do io
        JSON3.write(io, G |> to_dict)
    end
    return nothing
end # save
# ------------------------------------------------------------------------------
"""
    save(nzer::Normalizer{d}, filename::String) where {d}

Save the `Normalizer` `nzer` to a file `filename` in JSON format.
"""
function save(nzer::Normalizer{d}, filename::String) where {d}
    open(filename, "w") do io
        JSON3.write(io, nzer |> to_dict)
    end
end









# ------------------------------------------------------------------------------
"""
    parse_asg(di::Dict)::AdaptiveSparseGrid

Parse a dictionary `di` to an `AdaptiveSparseGrid`.
"""
function parse_asg(di::Dict)::AdaptiveSparseGrid
    d  = di["dim"]
    N = di["nodal"] |> length
    
    nodes = Node{d}[]
    nvals = NodeValue{d}[]
    for i in 1:N
        push!(nodes, Node{d}(
            di["levels"][i,:]  |> NTuple{d,Int},
            di["indices"][i,:] |> NTuple{d,Int}
        ))
        push!(nvals, NodeValue{d}(
            di["nodal"][i],
            di["hierarchical"][i]
        ))
    end

    return AdaptiveSparseGrid{di["dim"]}(
        Dictionary{Node{d}, NodeValue{d}}(nodes, nvals),
        di["depth"] |> Int,
        di["max_depth"] |> Int,
        di["max_levels"] |> NTuple{d,Int},
        di["rtol"] |> Float64,
        di["atol"] |> Float64,
        di["use_rtol"] |> Bool,
    )
end # parse_asg
# ------------------------------------------------------------------------------
"""
    parse_rsg(di::Dict)::RegularSparseGrid

Parse a dictionary `di` to a `RegularSparseGrid`.
"""
function parse_rsg(di::Dict)::RegularSparseGrid
    d  = di["dim"]
    N = di["nodal"] |> length
    
    nodes = Node{d}[]
    nvals = NodeValue{d}[]
    for i in 1:N
        push!(nodes, Node{d}(
            di["levels"][i,:]  |> NTuple{d,Int},
            di["indices"][i,:] |> NTuple{d,Int}
        ))
        push!(nvals, NodeValue{d}(
            di["nodal"][i],
            di["hierarchical"][i]
        ))
    end

    return RegularSparseGrid{di["dim"]}(
        Dictionary{Node{d}, NodeValue{d}}(nodes, nvals),
        di["max_levels"] |> NTuple{d,Int},
        di["max_depth"] |> Int,
    )
end # parse_rsg
# ------------------------------------------------------------------------------
"""
    parse_normalizer(di::Dict)::Normalizer

Parse a dictionary `di` to a `Normalizer`.
"""
function parse_normalizer(di::Dict)::Normalizer
    return Normalizer{di["dim"]}(
        di["lb"] |> NTuple{di["dim"],Float64},
        di["ub"] |> NTuple{di["dim"],Float64}
    )
end # parse_normalizer
# ------------------------------------------------------------------------------
"""
    parse(di::Dict)::Union{AbstractSparseGrid,Normalizer}

Parse a dictionary `di` to an `AbstractSparseGrid` or a `Normalizer`. The type
of the object is determined by the key "type" in the dictionary.
"""
function parse(di::Dict)::Union{AbstractSparseGrid,Normalizer}
    if !haskey(di, "type")
        throw(ArgumentError("Invalid dictionary"))
    elseif di["type"] == "AdaptiveSparseGrid"
        return parse_asg(di)
    elseif di["type"] == "RegularSparseGrid"
        return parse_rsg(di)
    elseif di["type"] == "Normalizer"
        return parse_normalizer(di)
    else
        throw(ArgumentError("Invalid dictionary"))
    end
end # parse










# ------------------------------------------------------------------------------
"""
    load(filename::String)::Dict

Read a JSON file `filename` and return the data structure dictionary.
"""
function load(filename::String)::Dict
    dat = JSON3.read(read(filename, String))
    # Symbol -> String key conversion due to the JSON3 package limitation
    return zip(
        string.(dat |> keys),
        dat |> values
    ) |> Dict
end # load



























