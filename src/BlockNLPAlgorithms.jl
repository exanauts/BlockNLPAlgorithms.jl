module BlockNLPAlgorithms
abstract type AbstractBlockNLPSolver end
abstract type AbstractBlockSolver end

struct ADMM <: AbstractBlockNLPSolver end
struct DualDecomposition <: AbstractBlockNLPSolver end
struct ProxADMM <: AbstractBlockNLPSolver end

using NLPModels
using Logging
using SolverCore
using BlockNLPModels
using LinearAlgebra
using SparseArrays
import Base: @kwdef
using MadNLP, MadNLPHSL

export AbstractBlockSolver # for designing custom solvers
export solve
export MadNLPSolver, IpoptSolver
include("block_solvers.jl")
include("options.jl")
include("main_solver.jl")
include("admm.jl")
include("prox_admm.jl")
include("dual_decomposition.jl")

end # module
