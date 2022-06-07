mutable struct ModelParams
    method::Type{<:AbstractBlockNLPSolver}

    opt::AbstractOptions
    start_time::Float64

    x::Vector{Float64} # primal solution
    y::Vector{Float64} # dual multipliers
    y_L::Vector{Float64} # bound dual multipliers
    y_U::Vector{Float64}

    # model
    nb::Integer
    model::AbstractBlockNLPModel
    init_blocks::Vector{AbstractNLPModel}
    init_obj_value::Float64
    full_model::AbstractNLPModel
    obj_value::Float64
    A::AbstractMatrix
    b::AbstractVector
    pr_res::AbstractVector # Ax-b
    jac::Tuple{AbstractVector,AbstractVector,AbstractVector}
    JTy::AbstractVector
    dl_res::AbstractVector

    tired::Bool
    converged::Bool
    max_subproblem_time::Vector{Float64}

    init_solver::AbstractVector
    results::Vector{BlockSolution}
end

"""
        solve(
            model::AbstractBlockNLPModel,
            method::AbstractBlockNLPSolver;
            options...
        )
Solves a `BlockNLPModel` with the specified `AbstractBlockNLPSolver` method.

# Arguments

- `m::AbstractBlockNLPModel`: identifier for the `BlockNLPModel` 
- `method::AbstractBlockNLPSolver`: solution algorithm
- `options`: solver options
"""
function solve(
    model::AbstractBlockNLPModel,
    method::Type{<:AbstractBlockNLPSolver};
    options...,
)
    init_model = initialize(model, method; options...)
    return solve!(init_model)
end

"""
        solve(
            model::AbstractBlockNLPModel,
            method::AbstractBlockNLPSolver;
            options...
        )
Solves a `BlockNLPModel` with the specified `AbstractBlockNLPSolver` method.

# Arguments

- `m::AbstractBlockNLPModel`: identifier for the `BlockNLPModel` 
- `method::AbstractBlockNLPSolver`: solution algorithm
- `options`: solver options
"""
function initialize(
    model::AbstractBlockNLPModel,
    method::Type{<:AbstractBlockNLPSolver};
    options...,
)
    start_time = time()

    nb = model.problem_size.block_counter
    options = Dict(options)

    # check if warm start primal-dual solutions are available 
    # otherwise initialize them with zero vectors
    if :primal_start ∈ keys(options)
        x = convert(Vector{Float64}, options[:primal_start])
        pop!(options, :primal_start)
    else
        x = zeros(Float64, model.problem_size.var_counter)
    end
    if :dual_start ∈ keys(options)
        y = convert(Vector{Float64}, options[:dual_start])
        pop!(options, :dual_start)
    else
        y = zeros(Float64, n_constraints(model))
    end

    opt = Options(primal_start = x, dual_start = y)
    set_options!(opt, options)
    initialized_blocks = initialize_blocks(model, method, opt)
    full_model = FullSpaceModel(model)

    # Initialize bound dual variables with zeros
    y_L = zeros(Float64, get_nvar(full_model))
    y_U = zeros(Float64, get_nvar(full_model))

    obj_value = obj(full_model, x)
    init_obj_value =
        sum(obj(initialized_blocks[i], x[model.blocks[i].var_idx]) for i = 1:nb)

    A = get_linking_matrix(model)
    b = get_rhs_vector(model)
    pr_res = similar(b) # to store ||Ax-b||
    mul!(pr_res, A, x)
    axpy!(-1.0, b, pr_res)

    dl_res = grad(full_model, x)

    # Get the J^T y product
    # To-do: find a more efficient way to store jacobian info
    jac = (
        jac_structure(full_model)[1],
        jac_structure(full_model)[2],
        jac_coord(full_model, x),
    )
    JTy = similar(dl_res)
    coo_prod!(jac[2], jac[1], jac[3], y, JTy)
    dl_res += JTy + y_L - y_U

    tired = false
    converged = false

    # initialize interior point solver for each block
    initialized_block_solvers = initialize_solver(opt.subproblem_solver, initialized_blocks)

    # initialize BlockSolution objects to store results for each block
    block_results = [
        BlockSolution(
            zeros(Float64, get_nvar(initialized_blocks[i])),
            0.0,
            0.0,
            zeros(Float64, get_ncon(initialized_blocks[i])),
            zeros(Float64, get_nvar(initialized_blocks[i])),
            zeros(Float64, get_nvar(initialized_blocks[i])),
        ) for i = 1:nb
    ]

    opt.verbosity > 0 && (@info log_row(
        Any[
            0,
            obj_value,
            init_obj_value,
            norm(pr_res),
            norm(dl_res),
            opt.step_size_min,
            time()-start_time,
            0.0,
        ],
    ))

    return ModelParams(
        method,
        opt,
        start_time,
        x,
        y,
        y_L,
        y_U,
        nb,
        model,
        initialized_blocks,
        init_obj_value,
        full_model,
        obj_value,
        A,
        b,
        pr_res,
        jac,
        JTy,
        dl_res,
        tired,
        converged,
        zeros(Float64, opt.max_iter),
        initialized_block_solvers,
        block_results,
    )
end

function initialize_blocks(model::AbstractBlockNLPModel, ::Type{ADMM}, opt::AbstractOptions)
    nb = model.problem_size.block_counter

    if opt.verbosity > 0
        @info log_header(
            [
                :iter,
                :objective,
                Symbol(:(aug_f(x))),
                :pr_inf,
                :dl_inf,
                :ρ,
                :elapsed_time,
                :max_block_time,
            ],
            [Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
        )
    end

    return [
        AugmentedNLPBlockModel(
            model.blocks[i],
            opt.dual_start[model.problem_size.con_counter+1:end],
            opt.step_size_min,
            get_linking_matrix(model),
            get_rhs_vector(model),
            opt.primal_start,
        ) for i = 1:nb
    ]
end

function initialize_blocks(
    model::AbstractBlockNLPModel,
    ::Type{DualDecomposition},
    opt::AbstractOptions,
)
    nb = model.problem_size.block_counter

    if opt.verbosity > 0
        @info log_header(
            [
                :iter,
                :objective,
                Symbol(:(dual_f(x))),
                :pr_inf,
                :dl_inf,
                :ρ,
                :elapsed_time,
                :max_block_time,
            ],
            [Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
        )
    end

    return [
        DualizedNLPBlockModel(
            model.blocks[i].problem_block,
            opt.dual_start[model.problem_size.con_counter+1:end],
            get_linking_matrix(model)[:, model.blocks[i].var_idx],
        ) for i = 1:nb
    ]
end

function initialize_blocks(
    model::AbstractBlockNLPModel,
    ::Type{ProxADMM},
    opt::AbstractOptions,
)
    nb = model.problem_size.block_counter

    if opt.verbosity > 0
        @info log_header(
            [
                :iter,
                :objective,
                Symbol(:(prox_aug_f(x))),
                :pr_inf,
                :dl_inf,
                :ρ,
                :elapsed_time,
                :max_block_time,
            ],
            [Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
        )
    end

    return [
        ProxAugmentedNLPBlockModel(
            model.blocks[i],
            opt.dual_start[model.problem_size.con_counter+1:end],
            opt.step_size_min,
            get_linking_matrix(model),
            get_rhs_vector(model),
            opt.primal_start,
            sparse(
                opt.proximal_penalty,
                model.blocks[i].meta.nvar,
                model.blocks[i].meta.nvar,
            ),
        ) for i = 1:nb
    ]
end
