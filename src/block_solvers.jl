struct MadNLPSolver <: AbstractBlockSolver
    options::Dict{Symbol,<:Any}
    function MadNLPSolver(; opts...)
        @eval using MadNLP
        new(Dict(opts))
    end
end

struct IpoptSolver <: AbstractBlockSolver
    options::Dict{Symbol,<:Any}
    function IpoptSolver(; opts...)
        @eval using Ipopt, NLPModelsIpopt
        new(Dict(opts))
    end
end

function optimize_block!(block::AbstractNLPModel, solver::IpoptSolver)
    result = ipopt(block; solver.options...)
end

function optimize_block!(block::AbstractNLPModel, solver::MadNLPSolver)
    result = madnlp(block; solver.options...)
end
