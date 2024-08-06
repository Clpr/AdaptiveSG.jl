# DEV-AdaptiveSG

The DEV version of `AdaptiveSG.jl`. Not registered yet.

## Description

An implementation of multi-linear Adaptive Sparse Grid (ASG) interpolation and 
a collection of utilities in solving high-dimensional HJB equations.


## Features

- Personalized recommendations for SG tasks
- Adaptive learning based on user feedback
- Interactive tutorials and code examples
- Collaboration and knowledge sharing

## Installation

To install and run the DEV-AdaptiveSG project, follow these steps:

2. Install Julia (>=1.9, 1.10.4 must work)
1. Create a new environment and install dependencies
    1. To create a new environment, check: `https://pkgdocs.julialang.org/v1/environments/`
    1. To install packages, check: `https://pkgdocs.julialang.org/v1/managing-packages/`
    1. `Plots.jl` is helpful in visualization
1. Clone the repository: `git clone https://github.com/Clpr/DEV-AdaptiveSG.git`
1. Start Julia with multi-threads (the training of ASG uses multi-threads)
    1. Use `julia --threads 4` to start a Julia session, where 4 can be replaced with other numbers
    1. Within a Julia session, use `Threads.nthreads()` to see the number of available threads
1. Import the main module to your code using `asg = include("./src/AdaptiveSG.jl")`

## File tree

- `src/`
    - `AdaptiveSG.jl`: the main module file
    - `asg/`: the training and updating methods of ASG
    - `common/`: common helpers
    - `datatypes/`: data structures and constructors
    - `io/`: I/O and vectorization methods
    - `operation/`: operations of data structures
- `test/`: testings & use cases (incomplete)


## Usage

TBD


## Key concepts

### Node{d}

A `Node{d}` struct is the parameteric representation of a $d$-dimensional grid node
in a 2-based hierarchical grid. It represents a $d$-dimensional point $x_{\mathbf{l,i}}$ in the hypercube
$[0,1]^d$ using a $d$-tuple of *levels* $\mathbf{l}$ and a $d$-tuple of *indices* $\mathbf{i}$.
The *depth* of a `Node{d}` is defined as $|\mathbf{l}|_{l_1}-d+1$.

- Manually define a node: `asg.Node{d}((l1,l2,...),(i1,i2,...))`
- Define an empty invalid node (usually as placeholders): `asg.Node{d}()`
- More node operations in: `src/operation/node.jl`

<font color=red>The node operations are the most technical part of this package.</font>

### NodeValue{d}

A `NodeValue{d}` is the tuple of nodal coefficient (origional function value) and
hierarchical coefficient linked to a `Node{d}`.

### AdaptiveSparseGrid{d}

An struct of multi-linear ASG interpolant for a function $f(x):[0,1]^d\to\mathbb{R}$.
It consists of:
- An ordered hashmap (`Dictionaries.jl`) that saves node-coefficients pairs. The underlying conceptual structure is an incomplete $2d$ tree
- Meta information of the interpolant

Check:
- Grid structure manupulation: `src/operation/gridstructure.jl`
- Training, updating and evaluation: `src/asg/`


### YellowPages{d}

A `YellowPages{d}` is a collection of by-dimnesion neighbor nodes for all the 
grid nodes in an ASG. This is kind of pre-conditioners to avoid repeating grid search.
Two types of neighbors are supported: 
- `:sparse`: actual neighbor of a grid node
- `:ghost`: ghost neighbor of a grid node


### Normalizer{d}

A `Normalizer{d}` defines a linear affine (normalization) from a $d$-dimensional rectangle region
$[lb_j,ub_j]^d,j=1,\dots,d$ to a hypercube $[0,1]^d$, and its inverse affine (de-normalization).
The `AdaptievSparseGrid{d}` does not handle such (de-)normalizations so the users need to
incoporate them in their functions to fit.


### LinearStencil{d}

A `LinearStencil{d}` consists of: a subset of nodes $\mathcal{P}\subseteq\mathcal{G}$ where $\mathcal{G}$ is the set of all grid nodes in a $d$-dimensional ASG interpolant; and a mapping from each nodes in $\mathcal{P}$ to scalar weights $w(p)$. Such a stencil represents the following linear operation:

$$
\sum_{p\in\mathcal{P}} w(p)\cdot f(p)
$$

where $f(p)$ is either the nodal coefficient or hierarchical coefficient at node $p$.

This struct supports arithmatic with scalars, and plus/minus with another `LinearStencil{d}`.
To apply the above operation and get the result, use `asg.apply()`





## License

This project is licensed under the [MIT License](LICENSE).


## Reference

- Schaab, A., & Zhang, A. (2022). Dynamic programming in continuous time with adaptive sparse grids. Available at SSRN 4125702.