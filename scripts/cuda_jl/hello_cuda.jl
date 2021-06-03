using CUDA
using BenchmarkTools

function cu_add_broadcast(x, y)
    CUDA.@sync y .+= x_d
end

N = 2^20
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

@btime cu_add_broadcast(x_d, y_d)
