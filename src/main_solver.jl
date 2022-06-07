function solve!(params::ModelParams)
    nb = params.nb # Number of blocks
    x = params.x
    y = params.y
    y_L = params.y_L
    y_U = params.y_U

    ρ = deepcopy(params.opt.step_size_min) # initialize ρ as the minimum allowed step size value
    μ = 10 # from Boyd's book (make options later)
    τ = 2

    iter_count = 0
    params.opt.update_scheme == :JACOBI && (temp_x = zeros(Float64, length(x)))
    elapsed_time = 0.0

    while !(params.converged || params.tired)
        iter_count += 1
        params.opt.update_scheme == :JACOBI && (fill!(temp_x, 0.0))
        max_iter_time = 0.0
        params.init_obj_value = 0.0 # reset to zero

        y_slice = @view y[params.model.problem_size.con_counter+1:end]

        for i = 1:nb
            params.method != DualDecomposition && update_primal!(params.init_blocks[i], x)
            update_dual!(params.init_blocks[i], y_slice)
            params.method != DualDecomposition &&
                params.opt.dynamic_step_size == true &&
                update_rho!(params.init_blocks[i], ρ)

            optimize_block!(params.init_solver[i], params.results[i])

            if params.opt.update_scheme == :GAUSS_SEIDEL
                x[params.model.blocks[i].var_idx] = params.results[i].solution
            elseif params.opt.update_scheme == :JACOBI
                temp_x[params.model.blocks[i].var_idx] = params.results[i].solution
            else
                error("Please choose the update scheme as :JACOBI or :GAUSS_SEIDEL")
            end

            params.results[i].elapsed_time > max_iter_time &&
                (max_iter_time = params.results[i].elapsed_time)
            params.init_obj_value += params.results[i].objective
            y[params.model.blocks[i].con_idx] = params.results[i].multipliers
            y_L[params.model.blocks[i].var_idx] = params.results[i].multipliers_L
            y_U[params.model.blocks[i].var_idx] = params.results[i].multipliers_U
        end
        params.opt.update_scheme == :JACOBI && (copyto!(x, temp_x))

        # update res
        mul!(params.pr_res, params.A, x)
        axpy!(-1.0, params.b, params.pr_res)

        y[params.model.problem_size.con_counter+1:end] +=
            params.opt.damping_param * ρ .* params.pr_res

        # update dual_res
        jac_coord!(params.full_model, x, params.jac[3])
        coo_prod!(params.jac[2], params.jac[1], params.jac[3], y, params.JTy)
        grad!(params.full_model, x, params.dl_res)
        params.dl_res += params.JTy - y_L + y_U
        elapsed_time = time() - params.start_time

        # update step-size
        if params.opt.dynamic_step_size == true
            norm(params.pr_res) > μ * norm(params.dl_res) && (ρ = ρ * τ)
            norm(params.dl_res) > μ * norm(params.pr_res) && (ρ = ρ / τ)

            ρ < params.opt.step_size_min && (ρ = params.opt.step_size_min)
            ρ > params.opt.step_size_max && (ρ = params.opt.step_size_max)
        end

        # evaluate stopping criteria
        params.tired =
            elapsed_time > params.opt.max_wall_time || iter_count >= params.opt.max_iter
        params.converged =
            norm(params.dl_res) <= params.opt.dl_feas_tol &&
            norm(params.pr_res) <= params.opt.pr_feas_tol

        params.obj_value = obj(params.full_model, x)

        params.opt.verbosity > 0 && (@info log_row(
            Any[
                iter_count,
                params.obj_value,
                params.init_obj_value,
                norm(params.pr_res),
                norm(params.dl_res),
                ρ,
                elapsed_time,
                max_iter_time,
            ],
        ))
        params.max_subproblem_time[iter_count] = max_iter_time
    end

    status = if params.converged
        :first_order
    elseif elapsed_time > params.opt.max_wall_time
        :max_time
    else
        :max_eval
    end

    return GenericExecutionStats(
        status,
        params.full_model,
        solution = x,
        objective = params.obj_value,
        iter = iter_count,
        primal_feas = norm(params.pr_res),
        dual_feas = norm(params.dl_res),
        elapsed_time = elapsed_time,
        multipliers = y,
        multipliers_L = y_L,
        multipliers_U = y_U,
        solver_specific = Dict(
            :max_subproblem_time => params.max_subproblem_time[1:iter_count],
        ),
    )
end
