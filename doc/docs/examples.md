# Examples

This section lists a collection of application examples where the package can work with.
We focus on examples particularly in economics research, which are hopefully inspiring for the research in other fields.


## Functional fixed point iteration

Consider two ideas:

1. Within a given tolerance, an interpolant can be approximately seen as the original function
1. With the adaption automatically determines the underlying grid structure, one can work with the whole function as one object without caring the grid.

The two ideas allow us to run the naive functional fixed point iteration for:

$$
f = F[f], f\mapsto \mathbb{R}
$$

where \(F\) is a fixed point functional.


(TBD)





## High-dimensional non-parametric regression

Fitting non-parametric regression model \(y_i = f(X_i) + \varepsilon_i\) is computation intensive as the dimensionality of the covariates \(X_i\) increasing.
The curse of dimensionality requires exponentially growing number of quantile knots when using local estimators such as kernel and spline estimators, as well as fast growing numebr of observations.

Now, ASG allows 


(TBD)




## Solving PDE with finite difference





(TBD)

