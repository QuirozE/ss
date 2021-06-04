using DrWatson
@quickactivate "ss"

using CUDA

function hello()
    @cuprintln("($(threadIdx().x), $(blockIdx().x)) Hello from CUDA.jl!!!")
    return
end

function main()
    @cuda hello()
    return
end

main()
