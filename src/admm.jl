"""
        admm(
            model::AbstractBlockNLPModel;
            options... 
        )
Solves a `BlockNLPModel` with the ADMM algorithm.

# Arguments

- `m::AbstractBlockNLPModel`: identifier for the `BlockNLPModel` 
- `options...`: options for the solver
"""
function solve!(::Type{ADMM}, params::ModelParams)

    nb = params.nb # Number of blocks
    x = params.x
    y = params.y
    ρ = params.opt.step_size

    iter_count = 0
    max_subproblem_time = zeros(Float64, params.opt.max_iter)
    temp_init_obj_value = 1e10
    obj_value = obj(params.full_model, x)
    init_obj_value = sum(obj(params.init_blocks[i], x[params.model.blocks[i].var_idx]) for i = 1:nb)
    elapsed_time = time() - params.start_time

    opt.verbosity > 0 && (@info log_row(
        Any[
            iter_count,
            obj_value,
            init_obj_value,
            norm(params.pr_res),
            norm(params.dl_res),
            params.opt.step_size,
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
            update_primal!(aug_blocks[i], x)
            update_dual!(aug_blocks[i], y_slice)

            optimize_block!(initialized_solver[i], results[i])

            if opt.update_scheme == :GAUSS_SEIDEL
                x[model.blocks[i].var_idx] = results[i].solution
            elseif opt.update_scheme == :JACOBI
                temp_x[model.blocks[i].var_idx] = results[i].solution
            else
                error("Please choose the update scheme as :JACOBI or :GAUSS_SEIDEL")
            end

            results[i].elapsed_time > max_iter_time && (max_iter_time = results[i].elapsed_time)
            aug_obj_value += results[i].objective
            y[model.blocks[i].con_idx] = results[i].multipliers
        end

        opt.update_scheme == :JACOBI && (copyto!(x, temp_x))
        # dual_res = grad(full_model, x) + 
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
