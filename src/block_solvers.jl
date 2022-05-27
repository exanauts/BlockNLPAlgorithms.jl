mutable struct BlockSolution
    solution::Vector{Float64}
    elapsed_time::Float64
    objective::Float64
    multipliers::Vector{Float64}
end

struct MadNLPSolver <: AbstractBlockSolver
    options::Dict{Symbol,<:Any}
    MadNLPSolver(; opts...) = new(Dict(opts))
end

struct IpoptSolver <: AbstractBlockSolver
    options::Dict{Symbol,<:Any}
    IpoptSolver(; opts...) = new(Dict(opts))
end

function initialize_solver end

function optimize_block! end
