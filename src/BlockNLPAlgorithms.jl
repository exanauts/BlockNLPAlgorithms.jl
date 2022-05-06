module BlockNLPAlgorithms
abstract type AbstractBlockSolver end

using NLPModels
using Logging
using SolverCore
using BlockNLPModels
using LinearAlgebra
import Base: @kwdef

export AbstractBlockSolver # for custom solvers
export admm, dual_decomposition
export MadNLPSolver, IpoptSolver

include("options.jl")
include("block_solvers.jl")
include("admm.jl")
include("dual_decomposition.jl")

end # module
