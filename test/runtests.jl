using SparseArrays
using LinearAlgebra
using BlockNLPModels
using BlockNLPAlgorithms
using JuMP
using NLPModelsJuMP
using Test

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
            @constraint(mpc_block, -0.01 <= x[1] <= 0.01)
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
# @testset "Solve using dual-decomposition" begin
#     solution = dual_decomposition(
#         block_mpc,
#         max_iter = 5000,
#         step_size = 0.4,
#         max_wall_time = 300.0,
#     )
# end
@testset "Solve using ADMM" begin
    solution =
        admm(block_mpc, primal_start = zeros(Float64, 2*N), max_iter = 5000, step_size = 0.4, max_wall_time = 50.0, update_scheme = "GAUSS_SEIDEL")
end
