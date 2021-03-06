module BlockNLPAlgorithms
abstract type AbstractBlockSolver end
abstract type AbstractBlockNLPSolver <: AbstractBlockSolver end

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

export AbstractBlockSolver # for designing custom solvers
export solve
export MadNLPSolver, IpoptSolver

include("block_solvers.jl")
include("options.jl")
include("solver_instance.jl")
include("main_solver.jl")

end # module
