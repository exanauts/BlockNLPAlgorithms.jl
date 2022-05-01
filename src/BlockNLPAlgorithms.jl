module BlockNLPAlgorithms

using NLPModels
using Logging
using SolverCore
using BlockNLPModels
using LinearAlgebra
using MadNLP
import Base: @kwdef

export admm, dual_decomposition

include("options.jl")
include("utils.jl")
include("admm.jl")
include("dual_decomposition.jl")

end # module
