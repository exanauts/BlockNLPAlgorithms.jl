"""
        admm(
            model::AbstractBlockNLPModel;
            kwargs... 
        )
Solves a `BlockNLPModel` with the ADMM algorithm.

# Arguments

- `m::AbstractBlockNLPModel`: identifier for the `BlockNLPModel` 
- `kwargs...`: options for the solver
"""
function admm(
    model::AbstractBlockNLPModel;
    option_dict::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    options...,
)
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
        y = zeros(Float64, n_constraints(m))
    end

    opt = Options(primal_start = x, dual_start = y)
    set_options!(opt, option_dict, options)

    tired = false
    converged = false

    if opt.verbosity > 0
        @info log_header(
            [:iter, :objective, :norm(Ax-b), :norm(λ), :elapsed_time, :max_block_time],
            [Int, Float64, Float64, Float64, Float64, Float64],
        )
    end
    obj_value = 0.0
    temp_obj_value = 1e10
    elapsed_time = 0.0

    # Get the linking constraints
    A = get_linking_matrix(model)
    b = get_rhs_vector(model)
    while !(converged || tired)
        iter_count += 1
        if opt.update_scheme == "JACOBI"
            temp_x = zeros(Float64, length(x))
        end
        max_iter_time = 0
        obj_value = 0.0 # reset to zero
        for i = 1:nb
            if opt.update_scheme == "GAUSS_SEIDEL"
                augmented_block = AugmentedNLPBlockModel(
                    model.blocks[i],
                    y[model.problem_size.con_counter+1:end],
                    opt.step_size,
                    A,
                    b,
                    x,
                )
                result = solve_block(augmented_block, opt.subproblem_solver, opt.verbosity)
                x[model.blocks[i].var_idx] = result.solution
            elseif opt.update_scheme == "JACOBI"
                augmented_block = AugmentedNLPBlockModel(
                    model.blocks[i],
                    y[model.problem_size.con_counter+1:end],
                    opt.step_size,
                    A,
                    b,
                    x,
                )
                result = solve_block(augmented_block, opt.subproblem_solver, opt.verbosity)
                temp_x[model.blocks[i].var_idx] = result.solution
            else
                error("Please choose the update scheme as either 'JACOBI' or 'GAUSS_SEIDEL'")
            end
            if result.elapsed_time > max_iter_time
                max_iter_time = result.elapsed_time
            end
            obj_value += result.objective
            y[model.blocks[i].con_idx] = result.multipliers
        end
        if opt.update_scheme == "JACOBI"
            x = deepcopy(temp_x)
        end
        y[model.problem_size.con_counter+1:end] .+=
            opt.damping_param * opt.step_size .* (A * x - b)

        elapsed_time = time() - start_time
        tired = elapsed_time > opt.max_wall_time || iter_count >= opt.max_iter
        converged =
            norm(obj_value - temp_obj_value)/abs(obj_value) <= opt.obj_conv_tol &&
            norm(A * x - b) <= opt.feas_tol
        temp_obj_value = deepcopy(obj_value)
        if opt.verbosity > 0
            @info log_row(
                Any[
                    iter_count,
                    obj_value,
                    norm(A * x - b),
                    norm(y[model.problem_size.con_counter+1:end]),
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
        FullSpaceModel(model),
        solution = x,
        objective = obj_value,
        iter = iter_count,
        primal_feas = norm(A * x - b),
        elapsed_time = elapsed_time,
        multipliers = y,
    )
end
