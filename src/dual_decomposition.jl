"""
        dual_decomposition(
            m::AbstractBlockNLPModel;
            kwargs... 
        )
Solves a `BlockNLPModel` with the ADMM algorithm.

# Arguments

- `m::AbstractBlockNLPModel`: identifier for the `BlockNLPModel` 
- `kwargs...`: options for the solver
"""
function dual_decomposition(
    m::AbstractBlockNLPModel;
    option_dict::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    kwargs...,
)
    start_time = time()
    nb = m.problem_size.block_counter # Number of blocks
    iter_count = 0

    # check if warm start primal-dual solutions are available 
    # otherwise initialize them with zero vectors
    if :primal_start ∈ keys(kwargs)
        x = Vector{Float64}(kwargs[:primal_start])
        pop!(kwargs, :primal_start)
    else
        x = zeros(Float64, m.problem_size.var_counter)
    end
    if :dual_start ∈ keys(kwargs)
        y = Vector{Float64}(kwargs[:dual_start])
        pop!(kwargs, :dual_start)
    else
        y = zeros(Float64, n_constraints(m))
    end

    opt = Options(primal_start = x, dual_start = y)
    set_options!(opt, option_dict, kwargs)

    tired = false
    converged = false

    if opt.verbosity > 0
        @info log_header(
            [:iter, :objective, :Ax_b, :λ, :elapsed_time, :max_block_time],
            [Int, Float64, Float64, Float64, Float64, Float64],
        )
    end
    obj_value = 0
    temp_obj_value = 1e10
    elapsed_time = 0

    # Get the linking constraints
    A = get_linking_matrix(m)
    b = get_rhs_vector(m)
    while !(converged || tired)
        iter_count += 1
        max_iter_time = 0
        obj_value = 0 # reset to zero
        for i = 1:nb
            dualized_block = DualizedNLPBlockModel(
                m.blocks[i].problem_block,
                y[m.problem_size.con_counter+1:end],
                A[:, m.blocks[i].var_idx],
            )
            result = solve_block(dualized_block, opt.subproblem_solver, opt.verbosity)
            x[m.blocks[i].var_idx] = result.solution

            if result.elapsed_time > max_iter_time
                max_iter_time = result.elapsed_time
            end
            obj_value += result.objective
            y[m.blocks[i].con_idx] = result.multipliers
        end
        y[m.problem_size.con_counter+1:end] .+=
            opt.damping_param * opt.step_size .* (A * x - b)

        elapsed_time = time() - start_time
        tired = elapsed_time > opt.max_wall_time || iter_count > opt.max_iter
        converged =
            norm(obj_value - temp_obj_value) <= opt.obj_conv_tol &&
            norm(A * sol - b) <= opt.feas_tol
        temp_obj_value = deepcopy(obj_value)
        if opt.verbosity > 0
            @info log_row(
                Any[
                    iter_count,
                    obj_value,
                    norm(A * x - b),
                    norm(y[m.problem_size.con_counter+1:end]),
                    elapsed_time,
                    max_iter_time,
                ],
            )
        end
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
        FullSpaceModel(m),
        solution = x,
        objective = obj_value,
        iter = iter_count,
        primal_feas = norm(A * x - b),
        elapsed_time = elapsed_time,
        multipliers = y,
    )
end
