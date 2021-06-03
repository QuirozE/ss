using CUDA

function hello()
    @cuprintln("($(threadIdx().x), $(blockIdx().x)) Hello from CUDA.jl!!!")
end

@cuda hello()
