struct MySolver <: AbstractBlockSolver
    function MySolver()
        @warn "This solver does not provide dual variable information."
        new()
    end
end

mutable struct PGDSolverInstance
    nlp::AbstractNLPModel
    pr_sol::AbstractArray
    grad::AbstractArray
end

function BlockNLPAlgorithms.initialize_solver(
    solver::MySolver,
    nlp_blocks::Vector{<:AbstractNLPModel},
)
    nb = length(nlp_blocks)

    return [
        PGDSolverInstance(
            nlp_blocks[i],
            zeros(Float64, get_nvar(nlp_blocks[i])),
            zeros(Float64, get_nvar(nlp_blocks[i])),
        ) for i = 1:nb
    ]
end

function BlockNLPAlgorithms.optimize_block!(
    init_sol::PGDSolverInstance,
    results::BlockNLPAlgorithms.BlockSolution,
)
    start_time = time()
    converged = false
    iter = 0
    alpha = 0.005 # step-size

    while !converged
        iter += 1
        # make a gradient step
        grad!(init_sol.nlp, init_sol.pr_sol, init_sol.grad)
        init_sol.pr_sol -= alpha .* init_sol.grad
        # project
        for i = 1:length(init_sol.pr_sol)
            if init_sol.pr_sol[i] >= init_sol.nlp.meta.uvar[i]
                init_sol.pr_sol[i] = init_sol.nlp.meta.uvar[i]
            elseif init_sol.pr_sol[i] <= init_sol.nlp.meta.lvar[i]
                init_sol.pr_sol[i] = init_sol.nlp.meta.lvar[i]
            end
        end
        # check convergence
        if norm(init_sol.grad) <= 1e-3 || iter >= 500
            converged = true
        end
    end

    results.solution = init_sol.pr_sol
    results.elapsed_time = time() - start_time
    results.objective = obj(init_sol.nlp, init_sol.pr_sol)
    results.multipliers = zeros(Float64, init_sol.nlp.meta.ncon)
    results.multipliers_L = zeros(Float64, init_sol.nlp.meta.nvar)
    results.multipliers_U = zeros(Float64, init_sol.nlp.meta.nvar)
end
