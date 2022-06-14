# BlockNLPAlgorithms.jl
This package implements several popular decomposition algorithms for block-structured nonlinear optimization problems (NLPs). It requires the NLPs to be implemented as `AbstractBlockNLPModels`; for more details on modeling block-structured NLPs, please refer to the documentation of [BlockNLPModels.jl](https://github.com/exanauts/BlockNLPModels.jl).

## Available Decomposition Algorithms
1. [Dual Decomposition Algorithm](http://www.seas.ucla.edu/~vandenbe/236C/lectures/dualdecomp.pdf)
2. [Alternating Direction Method of Multipliers or ADMM](https://stanford.edu/~boyd/admm.html)
3. [Proximal ADMM](http://alpha.math.uga.edu/~mjlai/papers/DLPYfinal.pdf)

## Installation
This package is currently under active development and has not been registered yet. However, to access the current code, one can enter the following command in Julia REPL:

```julia
]add "https://github.com/exanauts/BlockNLPAlgorithms.jl"
```

To confirm if the package was installed correctly, please run:
```julia
test BlockNLPAlgorithms
```
The test code generates and solves a small instance of a `BlockNLPModel` with both dual decomposition and ADMM, using [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl) to solve the subproblems.

## Quickstart Guide
To solve a `AbstractBlockNLPModel` with a decomposition algorithms implemented in this package, one can use the following method:

`solve(model::AbstractBlockNLPModel,method::Type{<:AbstractBlockNLPSolver};options...)`, where

- `model` is the block NLP
- `method` refers to the choice of the decomposition algorithms, user can choose one of the three available options:
    - `BlockNLPAlgorithms.DualDecomposition`
    - `BlockNLPAlgorithms.ADMM`
    - `BlockNLPAlgorithms.ProxADMM`
- `options` are the optional arguments to customize the behavior of the decomposition solver. Available solver options are listed in the next subsection.


### Solver options

`verbosity::Int`: This options sets the verbosity level of solver's output. Valid range is `[0,2]` with the default level being set to 1. 

`primal_start::Vector{Float64}`: Sets the initial guess values for the primal variable values. This gets initialized as a vector of zeros if not provided by the user.

`dual_start::Vector{Float64}`: Sets the initial guess values for the dual variable values. This gets initialized as a vector of zeros if not provided by the user.

`dynamic_step_size::Bool`: Determines whether the step size (or the penalty parameter) values is fixed or dynamic. This is set to `false` by default.

`step_size_min::Float64`: Determines the minimum and initial value of the step size (or the penalty parameter) used while solving the `BlockNLPModel`. In case `dynamic_step_size` is set to `false`, step size value remains fixed at this value.

`step_size_max::Float64`: Determines the maximum value of the step size (or the penalty parameter) used while solving the `BlockNLPModel`. This option does not do anything if `dynamic_step_size` is set to `false`.

`damping_param::Float64`: This option can be used to make the dual-update step size value different from the penalty parameter value in case of algorithms such as ADMM. More specifically, this option is implemented such that `step-size` = `damping_param` * `penalty parameter`. Default value is set to 1.

`update_scheme::Symbol`: Helps choose the primal update scheme during an iteration of the decomposition algorithm. User can choose between `:GAUSS_SEIDEL` (default) and `:JACOBI` update schemes.

`subproblem_solver::AbstractBlockSolver`: This option determines the solver used to solve the NLP subproblems obtained after decomposition. By default, subproblems are solved using `MadNLP.jl`.

`proximal_penalty::Union{AbstractArray,UniformScaling}`: This option can be used to set the weights of the proximal terms in case of the proximal ADMM algorithm. By default, the weights are set as `1e-3 .* I`, where `I` is the identity matrix.

`pr_feas_tol::Float64`: Termination tolerance for the primal feasibility criterion. The default value is `1e-4`.

`du_feas_tol::Float64`: Termination tolerance for the dual feasibility criterion. The default value is `1e-4`.

`max_iter::Int`: Maximum number of algorithm iterations. Solver terminates with exit status `:Maximum_Iterations_Exceeded` if primal and dual feasibility termination criteria tolerances are not met and decomposition algorithm's iteration count exceeds `max_iter`. The default value for this option is 100.

`max_wall_time::Float64`: Maximum wall time (in seconds) for the decomposition algorithm. Solver terminates with exit status `:Maximum_WallTime_Exceeded` if primal and dual feasibility termination criteria tolerances are not met and decomposition algorithm's iteration count exceeds `max_wall_time`. The default value for this option is 600 s.

### Steps for interfacing a subproblem solver
This package allows the user to provide use own subproblem solver. As an illustrative example, here we show the steps to interface this package with a projected gradient descent based solver for bound constrained quadratic programs (To-Do).

## Acknowledgements
This package's development was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.
