# AdaptiveSG.jl

`AdaptiveSG.jl` is a package for multi-linear adaptive sparse grid (ASG) interpolation. While the API is under active refinement, core functionality is fully operational.
To install this package, use `add https://github.com/Clpr/AdaptiveSG.jl.git` under the Pkg mode inside Julia.

Everyone is more than welcome to try and make feedback.

**Documentation**: [https://clpr.github.io/AdaptiveSG.jl/](https://clpr.github.io/AdaptiveSG.jl/)


## Requirement

`Julia>=1.9`. Earlier version may work but I have not tested them. The current version has a minimalism of pacakge dependency:

- `Dictionaries.jl`
- `StaticArrays.jl`




## Some Q&A

- **Question**: Why the initialization of `AdaptiveSparseGrid{D}` is separated from its training?
    - **Answer**: This package aims to allow low-level control of the grid structure. By separating the initialization and training, users can customize the grid before training the model. This degree of freedom is necessary in specific scenarios. If some users really want to put initialization and training in one place, a one-line function always works.
- **Question**: What is the performance?
    - **Answer**: Consider a 8-dimension function which has a shape of CRRA utility function. On my old desktop (Intel i7-9700), it takes about 116ms to train an RSG interpolant of $\approx$ 4000 nodes using single thread (however, the time cost of RSG locates mainly in the grid construction but not the training process). As for the ASG, it takes about 500ms to train $\approx$ 6000 nodes using 4 threads at a $1.5\%$ relative tolarence. Notice, with the same number of nodes, ASG is much more accurate than RSG. The evaluation is almost free.
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
    - **Answer**: Yes. One solution is training an RSG (no cost of trialing nodes) then adapting this RSG one or two steps forward as Schaab & Zhang (2022). I am still working on this and expect a new function `adapt!()` will be delivered in the future. The challenge here is to find a way that maintains some properties of the grid tree during the adaption.
- **Question**: Will the package support interpolations other than multi-linear?
    - **Answer**: Maybe. This is an ambitious idea. Theoretically, ASG can work with arbitrary interpolation methods. However, the current design is heavily based on some mathematical conclusions of multi-linear interpolation. I need to find an uniform framework that accommodates arbitrary degree of piecewise polynomial interpolations
- **Question**: Is multi-linear ASG a monotonic interpolation?
    - **Answer**: No. The node hierarchy destroys the monotonicity of the ordinary multi-linear interpolation. A recent paper has discussed this idea. If you require monotonic everywhere, then increasing the accuracy, or equivalently increasing the number of supporting nodes can mitigate this issue.
- **Question**: When I try solving HJB equations or other PDEs with your package, the finite difference method fails. Why?
    - **Answer**: When solving PDEs with finite difference and approximating the unknown function with an interpolant, the interpolation methods must satisfy specific conditions to keep the monotonicity of the scheme. Check my [blog post](https://clpr.github.io/blogs/post_241111.html) for a discussion about this issue.
- **Question**: Do you have a plan to support numerical quadrature for ASG?
    - **Answer**: Of course. This is on my schedule and has high priority. A trapezoid rule of one-dimensional partial integration has been implemented. The multi-dimensional integration is on my schedule
- **Question**: Does ASG support extrapolation?
    - **Answer**: By definition, no. Because the compact basis function forcely diminishes any evaluation at any out-of-the-box points to be 0. However, linear extrapolation is possible without changing the compact basis function. This is done by expanding the interpolation at the target outside point using Taylor expansion to the 1st order. Intuitively, evaluating an extrapolation query requires multiple interpolation evalautions, which is much more costly than the interpolation evaluation.


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