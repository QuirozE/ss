using DrWatson
@quickactivate "ss"

using CUDA
using BenchmarkTools


function bench_cu_broadcast(x, y)
    CUDA.@sync y .+= x_d
end

function cu_add(x, y)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx.x()
    offset = blockDim().x * gridDim().x

    for i in idx:offset:length(x)
        @inbounds x[i] += y[i]
    end

    return
end


function bench_cu_add(y, x)
    n_tr = 256
    n_blocks = ceil(Int, length(y)/n_tr)
    CUDA.@sync begin
        @cuda threads=n_tr blocks=n_blocks cu_add(y, x)
    end
end

N = 2^20
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

@btime bench_cu_broadcast(y_d, x_d)
@btime bench_cu_add(y_d, x_d)
