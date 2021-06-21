using DrWatson

@quickactivate "ss"

using Images

suc(z, c) = z*z + c

function in_julia(z; c=0, iters=200, threshold=1000)
    for _ in 1:iters
        z = suc(z, c)
        if abs(z) > threshold
            return false
        end
    end
    true
end

