# UTF-8
module AdaptiveSG

    using LinearAlgebra, SparseArrays
    using StaticArrays
    using Dictionaries
    import HDF5
    
    export Dictionary
    export SparseMatrixCSC, SparseVector

    include("datatypes/datatypes.jl") # data structures

    include("io/vectorize.jl") # grid structure vectorization
    # include("io/save.jl")    # save ASG to file
    # include("io/load.jl")    # load ASG from file

    include("common/math.jl")      # math (pure) functions
    include("common/common.jl")    # common helper functions
    include("common/phi.jl")       # basis function

    include("operation/node.jl")          # node & node value operations
    include("operation/stencil.jl")       # stencil arithmetic
    include("operation/gridstructure.jl") # grid structure modification etc.
    include("operation/yellowpages.jl")   # yellow pages operations
    include("operation/normalizer.jl")    # normalizer operations
    include("operation/finitediff.jl")    # finite difference stencil operations

    include("datatypes/constructors.jl") # (using some operations defined above)

    include("asg/train.jl")    # ASG (first-time) training
    include("asg/update.jl")   # ASG training after grid modification
    include("asg/evaluate.jl") # Interpolant evaluation

end # AdaptiveSG