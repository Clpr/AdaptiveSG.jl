# UTF-8
module AdaptiveSG

    using LinearAlgebra, SparseArrays
    using StaticArrays
    using Dictionaries
    
    export Dictionary
    export SparseMatrixCSC, SparseVector

    include("datatypes/datatypes.jl") # data structures

    include("io/vectorize.jl") # grid structure vectorization

    include("common/math.jl")      # math (pure) functions
    include("common/common.jl")    # common helper functions
    include("common/phi.jl")       # basis function

    include("operation/node.jl")          # node & node value operations
    include("operation/asg.jl")           # ASG grid structure modification etc.
    include("operation/rsg.jl")           # RSG grid structure modification etc.
    include("operation/normalizer.jl")    # normalizer operations
    include("operation/stdinterp.jl")     # API for standard interpolation lang

    include("datatypes/constructors.jl") # (using some operations defined above)

    include("asg/train.jl")    # ASG (first-time) training
    include("asg/update.jl")   # ASG training after grid modification
    include("asg/evaluate.jl") # Interpolant evaluation
    include("asg/integrate.jl") # Numerical integration of ASG/RSG interpolants

end # AdaptiveSG