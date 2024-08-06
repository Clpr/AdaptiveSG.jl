using Test
asg = include("../src/AdaptiveSG.jl")


@testset "Basic functionality: ASG training" begin
    include("train.jl")
end



@testset "Basic functionality: Stencil construction and arithmetic" begin
    include("stencil.jl")
end