function custom_solver end

function solve_block(
    block::AbstractNLPModel,
    solver::String,
    verbosity::Int;
    block_id::Union{Int,Nothing} = nothing,
)
    start_time = time()
    if solver == "IPOPT"
        result = ipopt(block, print_level = 4 * (verbosity > 1))
    elseif solver == "MadNLP"
        if verbosity > 1
            result = madnlp(block, print_level = MadNLP.INFO)
        else
            result = madnlp(block, print_level = MadNLP.WARN)
        end
    elseif solver == "USER-DEFINED"
        result = custom_solver(block, block_id)
    end
    return result
end
