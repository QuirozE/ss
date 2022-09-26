"""
Particle swarm is a nature inspired heuristic for optimization. It mimics the
patterns of flocks and fish schools to find global optimums.

At each step, each particle gets closer to its historical best and the global
best.
"""
module ParticleSwarm

export Particle, Swarm, move!, step!

function check_eq(mess, e0, es...)
    for (i, e) in enumerate(es)
        if e0 != e
            error(mess * "at element $i: $e0 ≠ $e")
        end
    end
end

abstract type Particle{T <: Real} end

"""
A particle has a position and a velocity.
"""
mutable struct AccParticle{T} <: Particle{T}
    pos :: Vector{T}
    velocity :: Vector{T}

    function AccParticle(pos)
        z = zeros(length(pos))
        new{T}(pos, z)
    end
end

# ```math
# x^{t+1} = (1 - \beta)x^{t} + \beta \mathbf{x}^{\star} + \alpha \epsilon
# ```
# where ``\mathbf{x}^{\star}`` is the current best position in the swarm, ``\beta``
# is a coefficient representing how attracted the particle is to the best
# particle, ``\alpha`` is a coefficient representing how likely is the particle
# to wander randomly, and ``\epsilon`` is a random vector from a ``N(0, 1)``
# distribution.
"""
Accelerated movement equation.
"""
function move!(p :: AccParticle, best, α, β, ϵ)
    check_eq(
        "In-congruent vector dimensions",
        length(p.pos), length(best),
        length(best), length(ϵ)
    )

    b = β .* best
    r = α .* ϵ
    p.pos = (1 - β) .* p.pos + b + r
end

move!(p :: AccParticle, best, α, β) = move!(
    p,
    best,
    α,
    β,
    rand(length(best)) .- 0.5
)

"""
A swarm of particles. It also keeps track of the current global best and common
particle values acceleration
"""
mutable struct Swarm{T<:Real}
    particles :: Vector{Particle{T}}
    best_pos :: Vector{T}
    objective_fn :: Function

    function Swarm(particles, cost)
        imin = argmin(cost.(particles))
        best_pos = pos[imin]
        new{T}(particles, best_pos, cost)
    end
end

end
