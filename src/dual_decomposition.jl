"""
        dual_decomposition(
            m::AbstractBlockNLPModel;
            options... 
        )
Solves a `BlockNLPModel` with the ADMM algorithm.

# Arguments

- `m::AbstractBlockNLPModel`: identifier for the `BlockNLPModel` 
- `options...`: options for the solver
"""
function dual_decomposition(model::AbstractBlockNLPModel; options...)
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
    full_model = FullSpaceModel(model)
    dual_blocks = [
        DualizedNLPBlockModel(
            model.blocks[i].problem_block,
            y[model.problem_size.con_counter+1:end],
            A[:, model.blocks[i].var_idx],
        ) for i = 1:nb
    ]

    tired = false
    converged = false

    if opt.verbosity > 0
        @info log_header(
            [
                :iter,
                :objective,
                Symbol(:(dual_f(x))),
                Symbol(:(Ax - b)),
                Symbol(:λ),
                :elapsed_time,
                :max_block_time,
            ],
            [Int, Float64, Float64, Float64, Float64, Float64, Float64],
        )
    end
    obj_value = obj(full_model, x)
    dual_obj_value = sum(obj(dual_blocks[i], x[model.blocks[i].var_idx]) for i = 1:nb)
    temp_dual_obj_value = 1e10
    elapsed_time = time() - start_time
    opt.verbosity > 0 && (@info log_row(
        Any[
            iter_count,
            obj_value,
            dual_obj_value,
            norm(A * x - b),
            norm(y[model.problem_size.con_counter+1:end]),
            elapsed_time,
            0.0,
        ],
    ))

    while !(converged || tired)
        iter_count += 1
        max_iter_time = 0.0
        dual_obj_value = 0.0 # reset to zero

        for i = 1:nb
            update_dual!(dual_blocks[i], y[model.problem_size.con_counter+1:end])
            result = optimize_block!(dual_blocks[i], opt.subproblem_solver)
            x[model.blocks[i].var_idx] = result.solution

            result.elapsed_time > max_iter_time && (max_iter_time = result.elapsed_time)
            dual_obj_value += result.objective
            y[model.blocks[i].con_idx] = result.multipliers
        end

        y[model.problem_size.con_counter+1:end] .+=
            opt.damping_param * opt.step_size .* (A * x - b)

        elapsed_time = time() - start_time

        # evaluate stopping criteria
        tired = elapsed_time > opt.max_wall_time || iter_count >= opt.max_iter
        converged =
            norm(dual_obj_value - temp_dual_obj_value) / abs(dual_obj_value) <=
            opt.obj_conv_tol && norm(A * x - b) <= opt.feas_tol

        obj_value = obj(full_model, x)
        temp_dual_obj_value = deepcopy(dual_obj_value)
        opt.verbosity > 0 && (@info log_row(
            Any[
                iter_count,
                obj_value,
                dual_obj_value,
                norm(A * x - b),
                norm(y[model.problem_size.con_counter+1:end]),
                elapsed_time,
                max_iter_time,
            ],
        ))
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
        primal_feas = norm(A * x - b),
        elapsed_time = elapsed_time,
        multipliers = y,
    )
end
