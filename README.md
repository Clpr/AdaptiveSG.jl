# AdaptiveSG.jl (Dev)

`AdaptiveSG.jl` is a development package for multi-linear adaptive sparse grid (ASG) interpolation. While the API is under active refinement, core functionality is fully operational. To experiment with this package, users may clone the repository and run `include("src/AdaptiveSG.jl")` to import the module.

Everyone is more than welcome to try and make feedback.

## Functionality & Features

- **ASG Interpolation Over a Hypercube**: Train an ASG interpolant within a hypercube or hyper-rectangle, evaluate the interpolant, and make structural modifications to the grid. Update interpolation coefficients to reflect these grid adjustments.

- **Data Structures for Multi-Dimensional PDEs**: Includes specialized data structures and functions that support the solution of multi-dimensional partial differential equations (PDEs) using finite difference and other numerical methods.

- **Serialization and Cross-Language Compatibility**: Provides serialization I/O functionality for model instances, enabling data to be parsed and utilized in other languages, including Python.

- **Numerical quadrature (in progress)**: Provides functions that perform numerical quadrature over ASG interpolants globally or marginally.

- **Detailed documentation**: Detailed documentation for each struct and functions that allows users to easily understand: what, how, and why. Whenever you have questions about anything, try the help documents.

## Requirement

`Julia>=1.9`. Earlier version may work but I have not tested them. The current dev version imports some 3rd-party packages:

- `Dictionaries.jl`
- `HDF5.jl` (optional, may be deprecated in the future)

**Steps to run**:

1. Install Julia and dependency packages. A virtual environment is recommended.
2. Clone this repository
3. Start Julia with multi-threads `julia --project=YOUR_ENV --threads NUM_THREADS` where `YOUR_ENV` is the name of your Julia environment, and `NUM_THREADS` is the number of threads to start Julia with (e.g. 4 for four cores). `AdaptiveSG.jl` work with multi-threads.
4. Import the main module e.g. `asg = include("PATH_TO_THIS_PROJECT/src/AdaptiveSG.jl")`

## Quick Reference

There are quick-start examples to train your first multi-linear ASG model.

```julia
asg = include("PATH_TO_THIS_PROJECT/src/AdaptiveSG.jl")

# display available structs and functions
varinfo(asg)

# train a 1-D sine function in range [0,1]
# and specify maxmium depth of the tree as 10
# the trainer receives a target function: AbstractVector -> Float64
G = asg.AdaptiveSparseGrid{1}(10, atol = 1E-2, use_rtol = false)
asg.train!(G, Xvec -> sin(Xvec[1]), printlevel = "iter")

# access the nodes and node values
display(G.nv)

# evaluate the interpolant at a random point
# passed as float vector
asg.evaluate(G, [2.0,])

# evaluate the interpolant at multiple points efficiently by defining a re-used call
geval(x) = asg.evaluate(G,[x])
geval.(rand(10))
```

The syntax `AdaptiveSparseGrid{D}` specifies the dimensionality `D` of the function, while users can choose either absolute tolerance `atol` or relative tolerance `rtol`. The option `use_rtol` controls which one to use. It is `false` by default. To visualize the interpolant:

```julia
import Plots as plt

# extract supporting nodes and function values as matrices
xnodes = asg.vectorize_x(G)
fnodes = asg.vectorize_nodal(G)

# draw the approximated sin(x) in range [0,1], and cp. w/ the true function
xtests = LinRange(0,1,100)
plt.plot(xtests, geval, label = "ASG")
plt.plot!(xtests, x -> sin(x), label = "True")
plt.scatter!(xnodes, fnodes, label = "supporting nodes")
```

When going to the multi-dimensional case, everything works similarly:

```julia
# train a 2-D exponential utility function in hypercube [0,1]^2
# but this time, use relative tolarence 0.1%
G = asg.AdaptiveSparseGrid{2}(15, rtol = 1E-3, use_rtol = true)
asg.train!(G, Xvec -> 2.0 + sum(sqrt.(Xvec)), printlevel = "iter")

# evaluate at point (0.114514, 0.1919810)
asg.evaluate(G, [0.114514, 0.1919810])

# if you prefer a re-used call
geval(x,y) = asg.evaluate(G,[x,y])

# extract supporting nodes
xnodes = asg.vectorize_x(G)
fnodes = asg.vectorize_nodal(G)

# visualize the surface
xtests = LinRange(0,1,100)
plt.surface(xtests, xtests, (x,y) -> geval(x,y), camera = (-30,30), xlabel = "x", ylabel = "y")
plt.scatter!(xnodes[:,1],xnodes[:,2], fnodes)
```

Either `AdaptiveSparseGrid{D}` or `train!` or `evaluate` are defined in hypercube $[0,1]^D$. To interpolate a function in a hyper rectangle, users can use `Normalizer{D}`:

```julia
#  - 3D function
#  - Hyper rectangle: [1,2]*[3,4]*[5,6]
G = asg.AdaptiveSparseGrid{3}(10, rtol = 1E-3, use_rtol = true)
nzer = asg.Normalizer{3}((1.0, 3.0, 5.0), (2.0, 4.0, 6.0)) # pass: min & max
asg.train!(
    G, 
    X01 -> begin
        X = asg.denormalize(X01, nzer)
        sin(X[1]) * cos(X[2]) * exp(X[3])
        end::Float64,
    printlevel = "iter"
)

# visualize a sliced surface
xtests = LinRange(1,2,100)
ytests = LinRange(3,4,100)
plt.surface(
    xtests, ytests, 
    (x,y) -> begin
        X01 = asg.normalize([x,y,5.5],nzer)
        asg.evaluate(G,X01)
    end, 
    camera = (-30,30),
    xlabel = "x", 
    ylabel = "y"
)
```

This package also implements regular sparse grid (RSG) which share the same syntax as ASG:

```julia
# similar to ASG, but one needs to manually specify:
# 1. maximum depth of the tree
# 2. maximum levels along each dimension
# However, RSG does not have the option to specify tolerance(s)

# Example: two-goods sqrt-root utility
G = asg.RegularSparseGrid{2}(5, (3,6))
asg.train!(G, Xvec -> sum(sqrt.(Xvec)), printlevel = "iter")

# evaluate at points
asg.evaluate(G, [0.23333,0.177013])
geval(x,y) = asg.evaluate(G,[x,y])
geval.(LinRange(0,1,20), LinRange(0,1,20))
```

The grid structure of an RSG instance is immutable and has different fields from ASG. Grid operations do not work with RSG types. To modify an RSG (e.g. adaption starting from an RSG), one may convert it to an equivalent ASG:

```julia
# needs to manually assign tolerance(s)
G2 = asg.convert2asg(G, rtol = 1E-3, use_rtol = true)
```

After training, one can always update the interpolation while keeping the same grid structure:

```julia
# first, train an ASG
G = asg.AdaptiveSparseGrid{2}(15, rtol = 1E-3, use_rtol = true)
asg.train!(G, Xvec -> 2.0 + sum(sqrt.(Xvec)), printlevel = "iter")

# manually add new empty nodes whose node values are NaN
asg.append!(G, [ asg.Node{2}((20,20),(1,1)), asg.Node{2}((20,20),(3,3)) ])

# update the ASG to fit the newly added nodes
# if you have access to the original function; recommended
asg.update!(G, Xvec -> 2.0 + sum(sqrt.(Xvec)) )

# or, udpate these nodes by interpolating them
# leads to almost-zero coefficients
asg.update!(G)

# then, keep the grid, but re-interpolate another function using the grid
asg.update_all!(G, Xvec -> 1.14514 + sum(Xvec .^ 1.5), printlevel = "iter")

# or, if you can only evaluate the new function at the existing nodes,
# then providing a mapping from these nodes to function values also works
# (use a begin block for readability)
begin
    # modify the grid structure before updating it
    # ...... e.g. asg.append!()
    
    # make an ordered Hash map, using the existing nodes
    # as key but mapping to Float64
    nv2 = asg.Dictionary( keys(G.nv), Vector{Float64}(undef, length(G.nv)) )
    
    # evalute the new function
    for node in keys(nv2)
        # node -> scalar x value (vector)
        Xvec = asg.get_x(node)
        
        nv2[node] = 1.14514 + sum(Xvec .^ 1.5)
    end
    
    # update the ASG interpolant
    asg.update_all!(G, nv2)
end
```

In later sections, I will introduce advanced functionalities esp. low-level grid structure modification (coming soon).

## Concepts

Many data structures in this package are rooted in the ASG mathematics introduced by Schiekofer (1998). Readers are highly encouraged to explore the theoretical papers e.g. Schaab & Zhang (2022) and Griebel (1998) for a deeper understanding, especially if they aim to develop their own packages using this package.

### Node{d}

A `Node{d}` struct is the parametric representation of a $d$-dimensional grid node in a 2-based hierarchical grid. It represents a $d$-dimensional point $x_{\mathbf{l,i}}$ in the hypercube $[0,1]^d$ using a $d$-tuple of *levels* $\mathbf{l}$ and a $d$-tuple of *indices* $\mathbf{i}$. The *depth* of a `Node{d}` is defined as $|\mathbf{l}|_{l_1}-d+1$. To define a hierarchical node (which not necessarily match the requirements of ASG):

```julia
# define a 3-D node
# dim1 (level,index) = (3,2)
# dim2 (level,index) = (4,3)
# dim3 (level,index) = (5,4)
# depth: (3+4+5) - 3 + 1 = 10
node = asg.Node{3}((3,4,5),(2,3,4))

# define an empty node
# all dims' levels = 0
# all dims' indices = -1
node = asg.Node{3}()

# check if a node is a valid node
asg.isvalid(node)

# initialize an empty array of nodes
Matrix{asg.Node{3}}(undef, 3, 4)
```

Node operations are the most technical part of this package. Please check `src/operation/node.jl` for a collection of node operations. Some examples:

- `isvalid()`: check if a given node instance satisfies all math properties
- `nodecmp()`: compare two nodes
- `nodecmp_except()`: compare two nodes dimension by dimension except one specific dimension
- `nodecmp_along()`: compare two nodes only along a specific dimension
- `get_x()`: get the real number value (vector) that is represented by a node
- `fraction2li()`: convert a fraction-representation of a one-dimension node to its level-index pair representation
- `perturb()`: modify a specific dimension of a node
- `sortperm(),sort(),sort!()`: sort a collection of nodes according to specific rules (useful in constructing triangle projection matrices)
- `get_child_left(),get_child_right(),get_parent()`: get the left/right child node or parent node of the grid tree along a given dimension
- `get_ghost_left(),get_ghost_right()`: get the ghost neighbor nodes of given depths
- `get_boundaryflag()`: return the integer flag that indicates if a node is on the boundary ($-1$: left boundary, $0$: interior, $1$: right boundary)
- `get_all_1d_nodal_nodes_in_order()`: list all 1-dimension nodes at a given level/depth
- `get_dist_along()`: computes the distance between two nodes along a given dimension
- `get_all_1d_regular_index()`: Return all the indices of the regular sparse nodes of a given level
- `glue()`: merges multiple “marginal” nodes to a higher dimension node

### NodeValue{d}

A `NodeValue{d}` is the tuple of nodal coefficient (original function value at the node) and hierarchical coefficient linked to a `Node{d}`. It can be validated by `isvalid()`.

### AdaptiveSparseGrid{d}

A struct of multi-linear ASG interpolant for a function $f(x):[0,1]^d\to\mathbb{R}$.
It consists of:

- An ordered Hash map (implemented by `Dictionaries.jl`) that saves node-coefficients pairs. The underlying conceptual structure is an incomplete $2d$ tree
- Meta information of the interpolant

Check `src/operation/asg.jl` for relative operations.

### RegularSparseGrid{d}

A struct of multi-linear RSG interpolant. Check `src/operation/rsg.jl`.

### Normalizer{d}

A `Normalizer{d}` defines a linear affine (normalization) from a $d$-dimensional rectangle region $[lb_j,ub_j]^d,j=1,\dots,d$ to a hypercube $[0,1]^d$, and its inverse affine (de-normalization). The `AdaptievSparseGrid{d}` does not handle such (de-)normalizations so the users need to incorporate them in their functions to fit.

### YellowPages{d}

A `YellowPages{d}` is a collection of by-dimension neighbor nodes for all the grid nodes in an ASG. This is kind of pre-conditioners to avoid repeating grid search. Two types of neighbors are supported: 
- `:sparse`: actual neighbor of a grid node
- `:ghost`: ghost neighbor of a grid node

Check `src/operation/yellowpages.jl`, and `src/datatypes/constructors.jl`


### LinearStencil{d} and HierarchicalBasisStencil{d}

A `LinearStencil{d}` consists of: a subset of nodes $\mathcal{P}\subseteq\mathcal{G}$ where $\mathcal{G}$ is the set of all grid nodes in a $d$-dimensional ASG interpolant; and a mapping from each nodes in $\mathcal{P}$ to scalar weights $w(p)$. Such a stencil represents the following linear operation:

$$
\sum_{p\in\mathcal{P}} w(p)\cdot f(p)
$$

where $f(p)$ is eithe the nodal coefficient or hierarchical coefficient at node $p$. A `HierarchicalBasisStencil{d}` is a specialized linear stencil in which $\mathcal{P}=\mathcal{G}$. This structure is useful in finite difference.

Both structs support regular arithmetic with scalars, and plus/minus with another `LinearStencil{d}`. To evaluate the above linear operation and get the scalar result, use `asg.apply()`. Check `src/opeartion/linear_stencil.jl` and `src/operation/hierar_stencil.jl`

## Advanced Examples

(Coming soon)

## Some Q&A

- **Question**: Why the initialization of `AdaptiveSparseGrid{D}` is separated from its training?
    - **Answer**: This package aims to allow low-level control of the grid structure. By separating the initialization and training, users can customize the grid before training the model. This degree of freedom is necessary in specific scenarios. If some users really want to put initialization and training in one place, a one-line function always works.
- **Question**: What is the performance?
    - **Answer**: On my old desktop (Intel i7-9700), it takes about 10-12 mins to train an 6-dimension ASG for a function of CRRA utility shape, using 4 threads. The evaluation is almost free.
- **Question**: At most how many nodes are allowed?
    - **Answer**: It depends on your machine also the capacity of the implementation of Hash map. On personal computer, I would recommend the dimensionality $\leq$ 7 to obtain a reasonable performance. For higher dimensionality, I am thinking about incorporating a SQLite engine to manage the grid tree (very far future!).
- **Question**: Which type of tolerance to use? Absolute or relative?
    - **Answer**: Either works but:
        - Absolute tolerance always works but you need to estimate the scale of the original function to trade off precision and time cost.
        - Relative tolerance is scale-free and usually faster and requiring less nodes than absolute tolerance. 
        - However, due to the definition of relative tolerance, it does not work well in the neighbor of $f(x)=0$. Even though the training algorithm automatically switches to absolute tolerance in this case, it still tends to put many annoyingly useless nodes there. So, if your original function $f$ crosses zero frequently, then use absolute tolerance. 
- **Question**: Do you have a method to delete specific node(s) from the grid?
    - **Answer**: Officially no. The current implementation emphasizes the reachability of the underlying tree. This property allows us to analytically visit children and parent nodes while free from undefined behaviors. Thus, the training algorithm only adds new nodes but never delete existing nodes. However, if you are 100% sure about what you are doing, then you can always manually `Dictionaries.delete!()` node(s) by manipulating the ordered Dictionary *after* the initial training.
- **Question**: Training an ASG from zero spends most of the time in trialing nodes that will not be accepted by the algorithm. Anyway to improve this?
    - **Answer**: Yes. One solution is training an RSG (no cost of trailing nodes) then adapting this RSG one or two steps forward as Schaab & Zhang (2022). I am still working on this and expect a new function `adapt!()` will be delivered in the future. The challenge here is to find a way that maintains some properties of the grid tree during the adaption.
- **Question**: Will the package support interpolations other than multi-linear?
    - **Answer**: Maybe. This is an ambitious idea. Theoretically, ASG can work with arbitrary interpolation methods. However, the current design is heavily based on some mathematical conclusions of multi-linear interpolation. I need to find an uniform framework that accommodates arbitrary degree of piecewise polynomial interpolations
- **Question**: Is multi-linear ASG a monotonic interpolation?
    - **Answer**: Yes. Exactly the same as standard multi-linear interpolation, multi-linear ASG does not overshoot.
- **Question**: When I try solving HJB equations or other PDEs with your package, the finite difference method fails. Why?
    - **Answer**: When solving PDEs with finite difference and approximating the unknown function with an interpolant, the interpolation methods must satisfy specific conditions to keep the monotonicity of the scheme. Check my [blog post](https://clpr.github.io/blogs/post_241111.html) for a discussion about this issue.

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