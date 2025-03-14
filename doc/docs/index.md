# AdaptiveSG.jl

This package implements multi-dimensional piecewise linear (aka multi-linear) interpolation of adaptive sparse grid (ASG).

[(Go to the project page)](https://github.com/Clpr/AdaptiveSG.jl)



## Installation

This package has not been registered in the general registry.
To import it as a module, run: `include("PROJECT_PATH/src/AdaptiveSG.jl")`;
To install it as a private package, run: `add "PROJECT_PATH/src/AdaptiveSG.jl"` in the Pkg mode.
In the future, one should use `add AdaptiveSG` in the Pkg mode.


## Problem setup

Consider a \(d\)-dimensional real function \(f(x):[0,1]^d\to\mathbb{R}\).
A generic interpolation function (*interpolant*) is defined as the following linear function:

$$
\hat{f} (x) := \sum_{i=1}^N \varphi_i(x) \cdot \theta_i
$$
where \(N\) is the number of *supporting nodes*; \(\varphi_i : [0,1]^{d}\to\mathbb{R}^{1\times N-1}\) is the *basis vector*; \(\theta_i\) is the *interpolation coefficient*.
The degree of the interpolation is \(N-1\).
An interpolant should approximate the original \(f(x)\) within a given error tolerance \(\varepsilon\).
To interpolate a function mapping from a hyper-rectangle domain \( \otimes_{j=1}^d [\underline{x}_j,\bar{x}_j] \), the normalization is required.

A multi-linear ASG interpolant falls into this generic framework. It is almost the same as a conventional multi-linear interpolation but:

- Using a hierarchy of nodes and coefficients, and residual fitting to effectively reduce the required number of nodes. (sparsity)
- Wisely choosing which nodes to add in the training process (adaption)


## Quick start

The package follows a pipeline:

1. `AdaptiveSparseGrid{d}`: Creating an instance of a multi-linear ASG interpolant of dimensionality \(d\). The interpolation tolerance type and value are specified.
1. `train!`: Training an instance by providing \(f(x)\).
1. `update!` and `update_all!`: In case of manually changing node(s), update the interpolation.
1. `evaluate`: Evaluate the interpolant at given points.
1. `integrate`: Integrate the interpolant along specific dimension(s).

### Fitting & evaluation in \([0,1]^d\)

Suppose we are going to fit the following function in a hypercube.

$$
f(x_1,x_2) := \sin(4x_1) + \cos(4x_2), (x_1,x_2) \in [0,1]^2
$$

To do this, we firstly create an instance of ASG interpolant and specify the tolerance:

```julia
import AdaptiveSG as asg

# absolute tolerance by default
G = asg.AdaptiveSparseGrid{2}(20, atol = 1E-3)

# relative tolerance
G = asg.AdaptiveSparseGrid{2}(20, rtol = 1E-3, use_rtol = true)
```

where the parametric parameter 2 indicates the dimensinoality.
The constructor accepts one position argument `max_depth` which specifies the
maximum depth of the node lattice, which represents the level of interpolation accuracy.
Here we set a numebr of 20 which is supposed to be enough for the algorithm to converge.
In the mathematical section, we talk more about the idea of node lattice, which can also be a tree conceptually with extra restrictions.


The constructor creates an empty instance without nodes. To fit \(f(x_1,x_2)\), we use `train!` function:

```julia
asg.train!(G, X -> sin(X[1]*4) + cos(X[2]*4), printlevel = "iter")
```

The `train!` function requires a function that receives a \(d\)-vector (here \(d=2\)) and returns a float number.
The keyword argument `printlevel` can be `iter`, `final` or `none`, which specifies at which level the information to print.

The training algorithm automatically test and decide which nodes to be added to the grid structure.
It ends when:

- Either adding a new node cannot improve the fitting given the tolerance type and value (converged)
- Or the node lattice has grown to the `max_depth` (pre-matured)

The choice of `max_depth` should trade off between: large enough to obtain a reasonable accuracy; small enough to avoid forever running. In many applications, it is expensive to evaluate the target function so one should be careful about the value of `max_depth`.

When an ASG interpolant is trained, it cannot be re-trained to fit another function because its grid structure is specialized for the current function. One should create another instance for another function. If there does exist a demand for reusing the same instance, please check the lower level operation guide.

After successfully training the ASG interpolant, one can evaluate its value at an arbitrary point in \([0,1]^2\):

```julia
yfit::Float64 = asg.evaluate(G, rand(2))
```

A callable wrapper could be helpful in some scenarios. It can be created by:

```julia
ffit(x1,x2) = asg.evaluate(G, [x1,x2])

# at 1 point
ffit(0.114514,0.1919810)

# at multiple points
xtest = rand(10,2)
ffit.(xtest[:,1], xtest[:,2])
```

Due to the theoretical property of node hierachy, the ASG interpolant does not
support extrapolation. Any point outside the \(d\)-hypercube will return a float zero.


On the current stage, the package supports integrating along one single dimension by providing at which point to do the integration:

```julia
xref = [0.5,0.5]

# Fixing x2 = 0.5, integrate along the 1st dimension
asg.integrate(G, xref, 1, xlim = (0.0, 1.0), nsamples = 10)

# weighted integration
asg.integrate(G, xref, 1, xlim = (0.0, 1.0), nsamples = 10, weight = rand(10))
```

where we provide a reference point `[0.5,0.5]`. The dimension to integrate of the reference point is ignored but we still need to specify a value there.
The keyword argument `weight` is useful in some applications (e.g. taking expectation),
while one needs to be careful that the provided `weight` is not normalized to sum-to-1.








### Hyper-rectangle domain

The vanila formula works with hyper cube \([0,1]^d\) domain to obtain the best performance.
However, in most cases, we work with a function that maps from a hyper rectangle domain \( \mathcal{X} := \otimes_{j=1}^d [\underline{x}_j,\bar{x}_j] \) where all lower and upper bounds are bounded.
Intuitively, a normalization or scaling is needed.
In this package, we specify the domain using data structure `Normalizer{d}`.

```julia
# specify domain: [0,1]^2 (default)
nzer = asg.Normalizer{2}()

# specify domain: Xrect := [-1,2] * [3,7]
nzer = asg.Normalizer{2}((-1.0, 3.0), (2.0, 7.0))
```

A normalizer can be used to affine an \(d\)-dimensional point between the hypercube \([0,1]^d\)
and the hyper-rectangle \(\mathcal{X}\):

```julia
# Xrect -> [0,1]^2
asg.normalize([1.0, 4.5], nzer)

# [0,1]^2 -> Xrect
asg.denormalize([0.5, 0.5], nzer)

# Xrect -> [0,1]^2, but only the 2nd dimension
asg.normalize(4.5, nzer, 2)

# [0,1]^2 -> Xrect, but only the 1st dimension
asg.denormalize(0.5, nzer, 1)
```

Now, to train the function in an alternaitve domain \(\mathcal{X}\),
one needs to explicitly normalize the target function before feeding it to `train!`. For example,

```julia
train!(
    G,
    X01 -> begin
        X = asg.denormalize(X01, nzer)
        sin(X[1]*4) + cos(X[2]*4)
    end
)
```

The other style, which is preferred by the authors, is defining the target function separately but only normalize it when passing it to `train!`:

```julia
f2fit(X) = sin(X[1]*4) + cos(X[2]*4)

train!(G, X01 -> f2fit(asg.normalize(X, nzer)))
```

This works better especially if each element of `X` has physical interpretations.

To evaluate the trained ASG interplant in the new domain of hyper rectangle \(\mathcal{X}\),
one can either:

```julia
# directly pass a point in the normalized [0,1]^2 space as the vanilla formula
asg.evalaute(G, [0.5,0.5])
```

Or, passing the point in the original space but with a normalizer:

```julia
asg.evaluate(G, [1.0, 4.5], nzer)
```

Of course, one can always do the denormalization explicitly:

```julia
asg.evaluate(G, asg.denormalize([1.0, 4.5], nzer))
```

To wrap up a callable object:

```julia
ffit(X) = asg.evalute(G, X, nzer)
```

The partial integration in the hyper rectangle domain is done by:

```julia
xref = [1.0, 4.5]

# integrate along the 1st dimension
asg.integrate(G, xref, 1, nzer, xlim = (-1.0, 2.0), nsample = 10)

# with weights
asg.integrate(G, xref, 1, nzer, xlim = (-1.0, 2.0), nsample = 10, weight = rand(10))
```






### Basis matrix, coefficients and stacking

In some applications, the interpolation coefficients and basis basis matrix are 
required explicitly.
The package provides API to extract them as vector and sparse matrix.

```julia
# number of the nodes
length(G)

# basis matrix at the nodes as a sparse lower-triangle matrix of length(G)^2
asg.basis_matrix(G)

# interpolation coefficient as a vector of length(G) elements
asg.interpcoef(G)
```

The stacked nodes in the hypercube can be extracted as:

```julia
# as a length(G) * d dense matrix
asg.vectorize_x(G)
```

There are more functions to stack other information of the interpolant.
Please check our lower level control section for more details.


The basis matrix can be evaluated at points other than the supporting nodes:

```julia
# evalute basis matrix/vector at point [.5,.5] in the normalized hypercube
asg.basis_matrix([0.5,0.5], G)

# evaluate basis matrix at multiple points in the normalized hypercube
xtest = rand(10,2)
asg.basis_matrix(xtest, G)
```

These functions provide another way of using this package.
Users that are experienced in other interpolation packages may find it familiar,
even though there can be significant performance loss.



### Regular sparse grid (RSG)

In addition to ASG, this package also implements regular sparse grid (RSG) which
generates anisotropic sparse grid multi-linear interpolants without adaption.
In some cases, shutting down adaption can significantly accelarate the training process at a cost of capturing local curvatures.

The `RegularSparseGrid{d}` shares exactly the same API and syntax as `AdaptiveSparseGrid{d}`,
while only some lower level control methods are not applicable.
Users who do not need those lower level control of the grid structure can safely ignore
the difference.

```julia
# create an RSG instance by specifying the max_depth of the whole node lattice
# and the the max_depth along each dimension
G = asg.RegularSparseGrid{2}(7, (5,5))

asg.train!(G, X -> sin(X[1]*4) + cos(X[2]*4), printlevel = "iter")

asg.evaluate(G, rand(2))

xref = [0.5,0.5]
asg.integrate(G, xref, 1, xlim = (0.0, 1.0), nsamples = 10)

nzer = asg.Normalizer{2}((-1.0, 3.0), (2.0, 7.0))

# Not like ASG, the RSG can be re-trained because the grid is fixed
f2fit(X) = sin(X[1]*4) + cos(X[2]*4)
train!(G, X01 -> f2fit(asg.normalize(X, nzer)))

asg.evaluate(G, [1.0, 4.5], nzer)

xref = [1.0, 4.5]
asg.integrate(G, xref, 1, nzer, xlim = (-1.0, 2.0), nsample = 10)

length(G)

asg.basis_matrix(G)
asg.interpcoef(G)

asg.vectorize_x(G)

asg.basis_matrix([0.5,0.5], G)
xtest = rand(10,2)
asg.basis_matrix(xtest, G)
```

To unlock the full power and API availabilty, one can convert it to an ASG interpolant:

```julia
G2 = asg.convert2asg(G)
```

The converted ASG interpolant inherits everything of the original RSG interpolant.
However, the tolerance, even if you specify them in `convert2asg`, is no longer consistent with what is really going on there. The value of tolerance is unreliable then.
















## Reference


- Schaab, A., & Zhang, A. (2022). Dynamic programming in continuous time with adaptive sparse grids. Available at SSRN 4125702.
- Schiekofer, T. (1998). *Die Methode der Finiten Differenzen auf d unnen Gittern zur L osung elliptischer und parabolischer partieller Di erentialgleichungen* (Doctoral dissertation, PhD thesis, Universit at Bonn).
- Griebel, M. (1998). Adaptive sparse grid multilevel methods for elliptic PDEs based on finite differences. *Computing*, *61*, 151-179.


This project is also inspired by:

- [AdaptiveSparseGrids.jl](https://github.com/jacobadenbaum/AdaptiveSparseGrids.jl) by Jacob Adenbaum
- [DistributedSparseGrids.jl](https://github.com/baxmittens/DistributedSparseGrids.jl) by Max Bittens and Daniel S. Katz
- [SGpp.jl](https://github.com/SGpp/SGpp) by SG++ development team


## License

This project is licensed under the [MIT License](LICENSE).
