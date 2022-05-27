using SparseArrays
using LinearAlgebra
using NLPModels
using NLsolve
using BlockNLPModels
using BlockNLPAlgorithms
using JuMP
using NLPModelsJuMP
using Test
using MadNLP

"""
Solves the following model predictive control problem:
```math
\\begin{aligned}
  \\min_{x \\in \\mathbb{R}, u \\in \\mathbb{R}} \\quad & \\sum\\limits_{i = 1}^{N-1} x_i^2 + u_i^2 + x_N^2 \\\\
  \\mathrm{subject \\, to} \\quad & x_{i+1} = x_i + u_i \\quad \\forall i \\in \\{ 1, \\ldots, N-1 \\} \\\\
  & x_1 = x0 \\\\
  & -1 \\leq u_i \\leq 1 \\quad \\forall i \\in \\{ 1, \\ldots, N-1 \\} \\\\
  & -0.01 \\leq x_N \\leq 0.01
\\end{aligned}
"""
A = fill(1, (1, 1));
B = fill(1, (1, 1));
x0 = 14
# Tuning matrices
Q = fill(1, (1, 1))
R = fill(1, (1, 1))

# Set the length of prediction horizon
N = 14

m = 1 # number of inputs
n = 1 # number of states
u_l = -1
u_u = 1
G = spzeros(N * n, N * (n + m))
F = zeros(N * n, n)

G[1:n, 1:m+n] = hcat(-B, I)
for i = 1:N-1
    G[i*n+1:i*n+n, (m+n)*i-n+1:(m+n)*i+n+m] = hcat(-A, -B, I)
end
F[1:n, 1:n] = A
# Initialize an empty block NLP model
block_mpc = BlockNLPModel()

# Fill the model with NLP blocks
for i = 1:2N
    if i % 2 != 0
        mpc_block = Model()
        @variables(mpc_block, begin
            u_l <= u[1:m] <= u_u
        end)
        @objective(mpc_block, Min, dot(u, R, u))
        nlp = MathOptNLPModel(mpc_block)
        add_block(block_mpc, nlp)
    else
        mpc_block = Model()
        @variable(mpc_block, x[1:n])
        if i == 2N
            set_lower_bound(x[1], -0.001)
            set_upper_bound(x[1], 0.001)
            # @constraint(mpc_block, -0.001 <= x[1] <= 0.001)
        end
        @objective(mpc_block, Min, dot(x, Q, x))
        nlp = MathOptNLPModel(mpc_block)
        add_block(block_mpc, nlp)
    end
end

# Add linking constraints all at once
links = Dict(1 => Matrix(G[:, 1:m]))
links[2] = Matrix(G[:, m+1:m+n])
count = 3
for i = 2:N
    global count
    for j = 1:2
        if count % 2 != 0
            links[count] = Matrix(G[:, (i-1)*(m+n)+1:(i-1)*(m+n)+m])
            count += 1
        else
            links[count] = Matrix(G[:, (i-1)*(m+n)+m+1:(i-1)*(m+n)+m+n])
            count += 1
        end
    end
end
add_links(block_mpc, N * n, links, F * [x0])

function BlockNLPAlgorithms.initialize_solver(solver::MadNLPSolver, nlp_blocks::Vector{<:AbstractNLPModel})
    nb = length(nlp_blocks)
    ips = Vector{MadNLP.InteriorPointSolver}(undef, nb)
    for i in 1:nb
        ips[i] = MadNLP.InteriorPointSolver(nlp_blocks[i]; solver.options...)
        MadNLP.initialize!(ips[i].kkt)
        MadNLP.initialize!(ips[i])
    end
    return ips
end

function BlockNLPAlgorithms.optimize_block!(initialized_block::MadNLP.InteriorPointSolver, results::BlockNLPAlgorithms.BlockSolution)
    # reset counters
    initialized_block.cnt = MadNLP.Counters(start_time = time())
    # solve
    optimal_solution = MadNLP.optimize!(initialized_block)
    # overwrite results
    field_names = fieldnames(typeof(results))
    for i in 1:length(field_names)
        setproperty!(results, field_names[i], getproperty(optimal_solution, field_names[i]))
    end
end

# dual_solution = dual_decomposition(
#     block_mpc,
#     max_iter = 5000,
#     step_size = 0.4,
#     max_wall_time = 300.0,
#     subproblem_solver = MadNLPSolver(print_level = MadNLP.WARN),
#     verbosity = 0,
# )

@profile admm_solution = solve(
    block_mpc,
    BlockNLPAlgorithms.ADMM,
    primal_start = zeros(Float64, 2 * N),
    max_iter = 5000,
    step_size = 0.5,
    max_wall_time = 100.0,
    update_scheme = :GAUSS_SEIDEL,
    verbosity = 1,
    subproblem_solver = MadNLPSolver(print_level = MadNLP.WARN),
)

pprof()

# @testset "Solve problem using off-the-shelf solvers" begin
#     @test round(admm_solution.objective) ≈ round(dual_solution.objective)
# end

# # Implement custom solver
# struct result
#     solution::Vector{Float64}
#     elapsed_time::Float64
#     objective::Float64
#     multipliers::Vector{Float64}
# end

# struct MySolver <: AbstractBlockSolver
#     options::Dict{Symbol,<:Any}
#     MySolver(; opts...) = new(Dict(opts))
# end

# function BlockNLPAlgorithms.optimize_block!(block::AbstractNLPModel, solver::MySolver)
#     start_time = time()
#     function f!(F, x)
#         F .= grad(block, x)
#     end

#     function j!(J, x)
#         J .=
#             sparse(hess_structure(block)[1], hess_structure(block)[2], hess_coord(block, x))
#     end

#     sol = nlsolve(f!, j!, zeros(Float64, block.meta.nvar)).zero
#     for i = 1:length(block.meta.nvar)
#         if sol[i] >= block.meta.uvar[i]
#             sol[i] = block.meta.uvar[i]
#         elseif sol[i] <= block.meta.lvar[i]
#             sol[i] = block.meta.lvar[i]
#         end
#     end
#     return result(
#         sol,
#         time() - start_time,
#         obj(block, sol),
#         zeros(Float64, block.meta.ncon),
#     )
# end

# my_solution = prox_admm(
#     block_mpc,
#     primal_start = zeros(Float64, 2 * N),
#     max_iter = 5000,
#     step_size = 0.4,
#     max_wall_time = 50.0,
#     update_scheme = :JACOBI,
#     subproblem_solver = MySolver(),
# )

# @testset "Solve problem using my solver" begin
#     @test ceil(my_solution.objective) ≈ round(admm_solution.objective)
# end
