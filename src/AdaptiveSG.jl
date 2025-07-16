# UTF-8
module AdaptiveSG

    using LinearAlgebra, SparseArrays
    import Printf: @printf, @sprintf

    include("math.jl")  # math (pure) helper functions & basis function ϕ
    include("class.jl") # data structures

    include("io.jl") # I/O operations
    
    include("train.jl") # training API


end # AdaptiveSG