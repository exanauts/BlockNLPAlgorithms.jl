module BlockNLPAlgorithms
abstract type AbstractBlockSolver end

using NLPModels
using Logging
using SolverCore
using BlockNLPModels
using LinearAlgebra
using SparseArrays
import Base: @kwdef

export AbstractBlockSolver # for custom solvers
export admm, dual_decomposition, prox_admm
export MadNLPSolver, IpoptSolver

include("options.jl")
include("block_solvers.jl")
include("admm.jl")
include("prox_admm.jl")
include("dual_decomposition.jl")

end # module
