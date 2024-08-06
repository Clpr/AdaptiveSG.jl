# item: the most plain 1-D case
# scenario: 
#  - 1D function
#  - receives a generic bracket-index-able object which covers SVector
#  - no normalization, defined in [0,1] domain
@test try
    G = asg.AdaptiveSparseGrid{1}(10, rtol = 1E-3)
    asg.train!(
        G, 
        Xvec -> sin(Xvec[1]), 
        printlevel = "none"
    )
    true
catch
    false
end

# item: the 1st most common case
# scenario:
#  - 3D function
#  - receives a generic bracket-index-able object which covers SVector
#  - no normalization, defined in [0,1]^3 domain
@test try
    G = asg.AdaptiveSparseGrid{3}(10, rtol = 1E-3)
    asg.train!(
        G,
        Xvec -> sin(Xvec[1]) * cos(Xvec[2]) * exp(Xvec[3]), 
        printlevel = "none"
    )
    true
catch
    false
end

# item: the 2nd most common case
# scenario:
#  - 3D function
#  - receives a generic bracket-index-able object which covers SVector
#  - normalization, [1,2]*[3,4]*[5,6] -> [0,1]^3 required
@test try
    G = asg.AdaptiveSparseGrid{3}(10, rtol = 1E-3)
    
    nzer = asg.Normalizer{3}((1.0, 3.0, 5.0), (2.0, 4.0, 6.0))
    
    function f2fit(X01 ; nzer::asg.Normalizer{3} = nzer)::Float64
        # [0,1]^d --> origional domain
        X = asg.denormalize(X01, nzer)
        return sin(X[1]) * cos(X[2]) * exp(X[3])
    end
    
    asg.train!(G, f2fit, printlevel = "none")

    true
catch
    false
end

































