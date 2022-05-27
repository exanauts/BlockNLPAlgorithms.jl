mutable struct ModelParams
    opt::AbstractOptions
    start_time::Float64

    x::Vector{Float64} # primal solution
    y::Vector{Float64} # dual solution

    # model
    nb::Integer
    model::AbstractBlockNLPModel
    init_blocks::Vector{AbstractNLPModel}
    full_model::AbstractNLPModel
    A::AbstractMatrix
    b::AbstractVector
    pr_res::AbstractVector # Ax-b
    jac::Tuple{AbstractVector, AbstractVector, AbstractVector}
    JTy::AbstractVector
    dl_res::AbstractVector

    tired::Bool
    converged::Bool

    init_solver::AbstractVector
    results::Vector{BlockSolution}
end

"""
        solve!(
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
function solve(model::AbstractBlockNLPModel, method::Type{<:AbstractBlockNLPSolver}; options...)
    solver = initialize(model, method; options)
    return solve!(method, solver)
end

function initialize(model::AbstractBlockNLPModel, method::Type{<: AbstractBlockNLPSolver}; options...)
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

    A = get_linking_matrix(model)
    b = get_rhs_vector(model)
    pr_res = similar(b) # to store ||Ax-b||
    mul!(pr_res, A, x)
    axpy!(-1.0, b, pr_res)

    dl_res = grad(full_model, x)
    
    # Get the J^T y product
    # To-do: find a more efficient way to store jacobian info
    jac = (jac_structure(full_model)[1], jac_structure(full_model)[2], jac_coord(full_model, x)[3])
    JTy = similar(dl_res)
    coo_prod!(jac[2], jac[1], jac[3], y, JTy)
    dl_res += JTy

    tired = false
    converged = false

    # initialize interior point solver for each block
    initialized_block_solvers = initialize_solver(opt.subproblem_solver, initialized_blocks)

    # initialize BlockSolution objects to store results for each block
    block_results = [BlockSolution(zeros(Float64, get_nvar(init_blocks[i])), 0., 0., zeros(Float64, get_ncon(init_blocks[i]))) for i in 1:nb]

    return ModelParams(opt, start_time, x, y, nb, model, initialized_blocks, full_model, A, b, pr_res, jac, JTy, dl_res, tired, converged, initialized_block_solvers, block_results)
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
            opt.step_size,
            get_linking_matrix(model),
            get_rhs_vector(model),
            opt.primal_start,
        ) for i = 1:nb
    ]
end

function initialize_blocks(model::AbstractBlockNLPModel, ::Type{DualDecomposition}, opt::AbstractOptions)
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

function initialize_blocks(model::AbstractBlockNLPModel, ::Type{ProxADMM}, opt::AbstractOptions)
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
            opt.step_size,
            get_linking_matrix(model),
            get_rhs_vector(b),
            opt.primal_start,
            sparse(
                opt.proximal_penalty,
                model.blocks[i].meta.nvar,
                model.blocks[i].meta.nvar,
            ),
        ) for i = 1:nb
    ]
end