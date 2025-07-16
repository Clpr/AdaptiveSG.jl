# AdaptiveSG.jl

`AdaptiveSG.jl` is a package for high-performance multi-linear adaptive sparse grid (ASG) interpolation.
To install this package, use `add AdaptiveSG` under the Pkg mode inside Julia.

Everyone is more than welcome to try and make feedback.

<!-- **Documentation**: [https://clpr.github.io/AdaptiveSG.jl/](https://clpr.github.io/AdaptiveSG.jl/) -->


## Requirement

`Julia>=1.9`.


## Usage

```julia
import AdaptiveSG as asg

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

# Dimensionality: 3
D = 3

# Create an empty R^D --> R sparse grid interpolant in a custom domain
G = asg.SparseGridInterpolant(D, lb = 1:D, ub = 2:D+1)

# Create an empty R^D --> R sparse grid interpolant in the hypercube [0,1]^D
G = asg.SparseGridInterpolant(D)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

# Example function to fit
f2fit(X) = (X .* 2π) .|> sin |> sum

# usage 1: train the interpolant using isotropic regular sparse grid (RSG)
# accuracy level: 7; isotropic max level per dimension: 5
asg.train!(
    G, 
    f2fit, 
    7, 
    5,
    verbose = true, # display training progress
)


# usage 2: anisotropic regular sparse grid (RSG)
asg.train!(
    G, 
    f2fit, 
    7, 
    Tuple(6:D+6-1), # anisotropic max levels per dimension
    verbose = true, # display training progress
)


# usage 3: local refinement/adaption after the initial RSG training, using
#          adaptive sparse grid (ASG)
#          max depth allowed: 13, tol = 3E-3, tolerance type: absolute
#          parallel threads enabled
asg.adapt!(
    G, f2fit, 13,
    verbose  = true,
    tol      = 3e-3,
    toltype  = :absolute, # or toltype = :relative
    parallel = true,      # use parallel threads
)



# ------------------------------------------------------------------------------
# Useful operations
# ------------------------------------------------------------------------------

# Get: size information
length(G) |> println # number of nodes
ndims(G)  |> println # dimension of the grid, = 4
size(G)   |> println # (number of nodes, dimension)


# Check: if a point locates in the grid domain
rand(D) in G
rand(D) ∈ G
rand(D) ∉ G

# Clamp a point to the grid domain
clamp(rand(D) .* 10, G)

# Draw random points in the grid domain
rand(G)      # a random point in the grid domain
rand(G, 10)  # 10 random points in the grid domain

# Create a uniform grid along the 2nd dimension with 10 points
LinRange(G, 2, 10)

# Transform a point between the grid domain and the unit hypercube [0,1]^4
let x0 = rand(D)

    # hypercube --> grid domain
    x1 = asg.scale(x0, 0.0, 1.0, to_lb = G.lb, to_ub = G.ub)
    x2 = asg.scale(
        x0, 
        zeros(length(x0)), 
        ones(length(x0)), 
        to_lb = G.lb, 
        to_ub = G.ub
    )

    # grid domain --> hypercube
    x3 = asg.scale(x1, G.lb, G.ub)
    x4 = asg.scale(x2, G.lb, G.ub, to_lb = 0.0, to_ub = 1.0)

    [x0 x1 x2 x3 x4]
end

# Stack all the supporting grid nodes into a N*D matrix
nodes = stack(G)


# Get the maximum levels of the grid in the grid hierarchy/tree
asg.maxlevels(G)



# ------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------

# Serialize the interpolant to a compatability-maximized dictionary
dat = asg.serialize(G)

# De-serialize the dictionary to an interpolant
G2 = asg.deserialize(dat)



# ------------------------------------------------------------------------------
# Translation to standard interpolation language
# ------------------------------------------------------------------------------

# get the interpolation/hierarchical coefficients
C = asg.coefficients(G)

# evaluate the basis function (as a basis vector) at a point
x = rand(D)
B = asg.basis(x, G)

# (manually) evaluate the interpolant at the point
sum(B .* C)

# evaluate the basis function at multiple points
X = rand(10, D)  # 10 points
B = asg.basis(X, G)

# (default) evaluate the basis function at all the grid nodes
B = asg.basis(G)
@assert size(B) == (length(G), length(G))

# construct: hierarchization and de-hierarchization matrices
H = asg.hierarchization_matrix(G)
E = asg.dehierarchization_matrix(G) # equivalent to `asg.basis(G)`



# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

# default: dimensional check applies, no extrapolation
x = rand(D)
G(x) 

# no dimensional check, (linear) extrapolation allowed
x = rand(D) .+ 1.0
G(
    x, 
    safe = false, 
    extrapolate = true
)



# ------------------------------------------------------------------------------
# A 2-D visualization example
# ------------------------------------------------------------------------------

# mimics: exponential utility function which needs adpation as x --> 0
f2fit(X) = -1 / (0.1 + sum(X))

G = asg.SparseGridInterpolant(2)
asg.train!(
    G, 
    f2fit, 
    4, 
    (3,8),
    verbose = true, # display training progress
)
asg.adapt!(
    G, f2fit, 20,
    verbose = true,
    tol     = 1e-5,
    toltype = :absolute,
    parallel = false,
)


import Plots as plt

# fig: approximation errors
fig = plt.surface(
    LinRange(G, 1, 100),
    LinRange(G, 2, 100),
    (x,y) -> G([x,y]) - f2fit([x,y]),
    camera = (-30,30),
    alpha = 0.5,
    colorbar = false,
    legend = false,
)
fig

# fig: grid nodes distribution
fig = plt.surface(
    LinRange(G, 1, 100),
    LinRange(G, 2, 100),
    (x,y) -> G([x,y]),
    camera = (-30,30),
    alpha = 0.5,
    colorbar = false,
    legend = false,
)
plt.scatter!(
    fig,
    (stack(G) |> eachcol)...,
    G.fvals,
)
fig
```

## Future plan

- Adds practical testing cases
- Supports sparse quadrature/integration
- Supports polynomial basis

## License

This project is licensed under the [MIT License](LICENSE).

## Similar Projects

- [AdaptiveSparseGrids.jl](https://github.com/jacobadenbaum/AdaptiveSparseGrids.jl) by Jacob Adenbaum
- [DistributedSparseGrids.jl](https://github.com/baxmittens/DistributedSparseGrids.jl) by Max Bittens and Daniel S. Katz
- [SGpp.jl](https://github.com/SGpp/SGpp) by SG++ development team

## Reference

- Schaab, A., & Zhang, A. (2022). Dynamic programming in continuous time with adaptive sparse grids. Available at SSRN 4125702.
- Schiekofer, T. (1998). *Die Methode der Finiten Differenzen auf d unnen Gittern zur L osung elliptischer und parabolischer partieller Di erentialgleichungen* (Doctoral dissertation, PhD thesis, Universit at Bonn).
- Griebel, M. (1998). Adaptive sparse grid multilevel methods for elliptic PDEs based on finite differences. *Computing*, *61*, 151-179.