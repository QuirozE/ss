"""
Particle swarm is a nature inspired heuristic for optimization. It mimics the
patterns of flocks and fish schools to find global optimums.

At each step, each particle gets closer to its historical best and the global
best.
"""
module ParticleSwarm

export Particle, AccParticle, Swarm, move!, step!

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
    pos::Vector{T}

    function AccParticle(pos::Vector{T}) where {T <: Real}
        new{T}(pos)
    end
end

function Base.iterate(p::AccParticle, state=1)
    if state >= length(p.pos)
        nothing
    else
        (p.pos[state], state + 1)
    end
end

Base.length(p::AccParticle) = Base.length(p.pos)

Base.getindex(p::AccParticle, i) = Base.getindex(p.pos, i)

"""
Accelerated movement equation.
"""
function move!(p::AccParticle, best::AccParticle, α, β, ϵ)
    check_eq(
        "In-congruent vector dimensions",
        length(p.pos), length(best.pos),
        length(best.pos), length(ϵ)
    )

    b = β .* best.pos
    r = α .* ϵ
    p.pos = (1 - β) .* p.pos + b + r
end

move!(p, best, α, β) = move!(
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
mutable struct Swarm
    particles
    best_particle
    objective_fn
    α
    β

    function Swarm(particles::Vector, cost; α=0.2, β=0.5)
        imin = argmin(cost.(particles))
        best_pos = particles[imin]
        new(particles, best_pos, cost, α, β)
    end
end

function Swarm(dims::Integer, cost; num_particles=10, type=apso, kwargs...)
    if type == apso
        particles = [AccParticle(rand(dims)) for _ in 1:num_particles]
        Swarm(particles, cost, kwargs...)
    end
end

@enum ParticleType begin
    apso
end

function step!(swarm)
    for (idx, particle) in enumerate(swarm.particles)
        move!(particle, swarm.best_particle, swarm.α, swarm.β)
    end

    imin = argmin(swarm.objective_fn.(swarm.particles))
    swarm.best_particle = swarm.particles[imin]
end

function pso(cost, dims; min_change=0.1, kwargs)
    swarm = Swarm(cost, dims, kwargs)
end

end # module
