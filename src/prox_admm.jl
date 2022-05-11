"""
        prox_admm(
            model::AbstractBlockNLPModel;
            options... 
        )
Solves a `BlockNLPModel` with the ADMM algorithm.

# Arguments

- `m::AbstractBlockNLPModel`: identifier for the `BlockNLPModel` 
- `options...`: options for the solver
"""
function prox_admm(model::AbstractBlockNLPModel; options...)
    start_time = time()
    nb = model.problem_size.block_counter # Number of blocks
    iter_count = 0

    options = Dict(options)
    # check if warm start primal-dual solutions are available 
    # otherwise initialize them with zero vectors
    if :primal_start ∈ keys(options)
        x = Vector{Float64}(options[:primal_start])
        pop!(options, :primal_start)
    else
        x = zeros(Float64, model.problem_size.var_counter)
    end
    if :dual_start ∈ keys(options)
        y = Vector{Float64}(options[:dual_start])
        pop!(options, :dual_start)
    else
        y = zeros(Float64, n_constraints(model))
    end

    opt = Options(primal_start = x, dual_start = y)
    set_options!(opt, options)

    # Prepare to start the solution algorithm
    A = get_linking_matrix(model)
    b = get_rhs_vector(model)
    res = similar(b) # to store ||Ax-b||
    mul!(res, A, x)
    axpy!(-1.0, b, res)

    full_model = FullSpaceModel(model)
    prox_aug_blocks = [
        ProxAugmentedNLPBlockModel(
            model.blocks[i],
            y[model.problem_size.con_counter+1:end],
            opt.step_size,
            A,
            b,
            x,
            sparse(
                opt.proximal_penalty,
                model.blocks[i].meta.nvar,
                model.blocks[i].meta.nvar,
            ),
        ) for i = 1:nb
    ]

    tired = false
    converged = false

    if opt.verbosity > 0
        @info log_header(
            [
                :iter,
                :objective,
                Symbol(:(aug_f(x))),
                Symbol(:(Ax - b)),
                Symbol(:λ),
                :elapsed_time,
                :max_block_time,
            ],
            [Int, Float64, Float64, Float64, Float64, Float64, Float64],
        )
    end
    obj_value = obj(full_model, x)
    aug_obj_value = sum(obj(prox_aug_blocks[i], x[model.blocks[i].var_idx]) for i = 1:nb)
    temp_aug_obj_value = 1e10
    elapsed_time = time() - start_time
    max_subproblem_time = zeros(Float64, opt.max_iter)

    opt.verbosity > 0 && (@info log_row(
        Any[
            iter_count,
            obj_value,
            aug_obj_value,
            norm(res),
            norm(y[model.problem_size.con_counter+1:end]),
            elapsed_time,
            0.0,
        ],
    ))
    opt.update_scheme == :JACOBI && (temp_x = zeros(Float64, length(x)))

    while !(converged || tired)
        iter_count += 1
        opt.update_scheme == :JACOBI && (fill!(temp_x, 0.0))
        max_iter_time = 0.0
        aug_obj_value = 0.0 # reset to zero

        y_slice = @view y[model.problem_size.con_counter+1:end]
        for i = 1:nb
            update_primal!(prox_aug_blocks[i], x)
            update_dual!(prox_aug_blocks[i], y_slice)

            if opt.update_scheme == :GAUSS_SEIDEL
                result = optimize_block!(prox_aug_blocks[i], opt.subproblem_solver)
                x[model.blocks[i].var_idx] = result.solution
            elseif opt.update_scheme == :JACOBI
                result = optimize_block!(prox_aug_blocks[i], opt.subproblem_solver)
                temp_x[model.blocks[i].var_idx] = result.solution
            else
                error("Please choose the update scheme as :JACOBI or :GAUSS_SEIDEL")
            end

            result.elapsed_time > max_iter_time && (max_iter_time = result.elapsed_time)
            aug_obj_value += result.objective
            y[model.blocks[i].con_idx] = result.multipliers
        end

        opt.update_scheme == :JACOBI && (copyto!(x, temp_x))

        # update res
        mul!(res, A, x)
        axpy!(-1.0, b, res)

        y[model.problem_size.con_counter+1:end] += opt.damping_param * opt.step_size .* res

        elapsed_time = time() - start_time

        # evaluate stopping criteria
        tired = elapsed_time > opt.max_wall_time || iter_count >= opt.max_iter
        converged =
            abs(aug_obj_value - temp_aug_obj_value) / abs(aug_obj_value) <=
            opt.obj_conv_tol && norm(res) <= opt.feas_tol

        obj_value = obj(full_model, x)
        temp_aug_obj_value = deepcopy(aug_obj_value)

        opt.verbosity > 0 && (@info log_row(
            Any[
                iter_count,
                obj_value,
                aug_obj_value,
                norm(res),
                norm(y_slice),
                elapsed_time,
                max_iter_time,
            ],
        ))
        max_subproblem_time[iter_count] = max_iter_time
    end

    status = if converged
        :acceptable
    elseif elapsed_time > opt.max_wall_time
        :max_time
    else
        :max_eval
    end

    return GenericExecutionStats(
        status,
        full_model,
        solution = x,
        objective = obj_value,
        iter = iter_count,
        primal_feas = norm(res),
        elapsed_time = elapsed_time,
        multipliers = y,
        solver_specific = Dict(:max_subproblem_time => max_subproblem_time[1:iter_count]),
    )
end
