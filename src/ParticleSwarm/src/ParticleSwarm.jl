"""
Particle swarm is a nature inspired heuristic for optimization. It mimics the
patterns of flocks and fish schools to find global optimums.

At each step, each particle gets closer to its historical best and the global
best.
"""
module ParticleSwarm

export Particle, AccParticle, Swarm, move!, step!, pso

function check_eq(mess, e0, es...)
    for (i, e) in enumerate(es)
        if e0 != e
            error(mess * "at element $i: $e0 ≠ $e")
        end
    end
end

abstract type Particle{T <: Real} end

"""
A particle has a position.
"""
mutable struct AccParticle{T} <: Particle{T}
    pos::Vector{T}
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

# stub because accelerated particles don't store their best position
update_best!(p::AccParticle, _) = true

mutable struct BoolParticle <: Particle{Bool}
    pos
    vel
    best_pos

    function BoolParticle(pos::Vector{Bool})
        new(pos, zeros(length(pos)), pos)
    end
end

function Base.iterate(p::BoolParticle, state=1)
    if state >= length(p.pos)
        nothing
    else
        (p.pos[state], state + 1)
    end
end

Base.length(p::BoolParticle) = Base.length(p.pos)

Base.getindex(p::BoolParticle, i) = Base.getindex(p.pos, i)

function move!(p::BoolParticle, best::BoolParticle, α, β, ϵ)
    vel_p_best = α .* (p.pos - p.best_pos)
    vel_best = β .* (p.pos - best.pos)
    p.vel = p.vel + ϵ .* (vel_p_best + vel_best)

    probs = sigmoid.(p.vel)
    p.pos = rnd_swap.(probs)
end

sigmoid(x) = 1/(1 + ℯ^(-x))

rnd_swap(p) = rand() < p

function update_best!(p::BoolParticle, cost)
    if cost(p.pos) < cost(p.best_pos)
        p.best_pos = p.pos
    end
end

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

    function Swarm(particles::Vector, cost; α=0.2, β=0.5, kwargs...)
        imin = argmin(cost.(particles))
        best_pos = particles[imin]
        new(particles, best_pos, cost, α, β)
    end
end

function Swarm(dims::Integer, cost; num_particles=10, type=apso, range=0:0.1:1, α = 0.2, β = 0.5, kwargs...)
    if type == apso
        particles = [AccParticle(rand(range, dims)) for _ in 1:num_particles]
        Swarm(particles, cost, α = α, β = β, kwargs...)
    elseif type == bpso
        particles = [BoolParticle(rand(Bool, dims)) for _ in 1:num_particles]
        Swarm(particles, cost, α = α, β = β, kwargs...)
    end
end

@enum ParticleType begin
    apso
    bpso
end

function step!(swarm)
    for particle in swarm.particles
        move!(particle, swarm.best_particle, swarm.α, swarm.β)
        update_best!(particle, swarm.objective_fn)
    end

    imin = argmin(swarm.objective_fn.(swarm.particles))
    swarm.best_particle = swarm.particles[imin]
end

function pso(cost, dims; steps = 20, kwargs...)
    swarm = Swarm(dims, cost, kwargs...)
    for i in 1:steps
        step!(swarm)
    end
    swarm.best_particle
end

end # module
