struct MadNLPSolver <: AbstractBlockSolver
    options::Dict{Symbol,<:Any}
    MadNLPSolver(; opts...) = new(Dict(opts))
end

struct IpoptSolver <: AbstractBlockSolver
    options::Dict{Symbol,<:Any}
    IpoptSolver(; opts...) = new(Dict(opts))
end

function optimize_block! end