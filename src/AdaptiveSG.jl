# UTF-8
module AdaptiveSG

    using LinearAlgebra, SparseArrays
    import Printf: @printf, @sprintf

    import JSON3


    include("math.jl")  # math (pure) helper functions & basis function ϕ
    include("class.jl") # data structures

    include("io.jl") # I/O operations
    
    include("train.jl") # training API



    # include("datatypes/datatypes.jl") # data structures

    # include("io/vectorize.jl") # grid structure vectorization
    # include("io/serialize.jl") # grid structure serialization

    # include("common/math.jl")      # math (pure) functions
    # include("common/common.jl")    # common helper functions
    # include("common/phi.jl")       # basis function

    # include("operation/node.jl")          # node & node value operations
    # include("operation/normalizer.jl")    # normalizer operations
    # include("operation/asg.jl")           # ASG grid structure modification etc.
    # include("operation/rsg.jl")           # RSG grid structure modification etc.
    # include("operation/stdinterp.jl")     # API for standard interpolation lang

    # include("datatypes/constructors.jl") # (using some operations defined above)

    # include("asg/train.jl")    # ASG (first-time) training
    # include("asg/update.jl")   # ASG training after grid modification
    # include("asg/extrapolate.jl") # ASG (linear) extrapolation
    # include("asg/evaluate.jl") # Interpolant evaluation
    # include("asg/diff.jl")     # Gradient approximation
    # include("asg/integrate.jl") # Numerical integration of ASG/RSG interpolants

    # include("wrapper.jl") # A convenient wrapper (will replace the current API)

end # AdaptiveSG