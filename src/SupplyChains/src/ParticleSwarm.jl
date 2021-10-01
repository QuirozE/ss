"""
Particle swarm is a nature inspierd heuristic for optimization. It mimics the
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

"""
A particle has a position and a momentum. It also remebers its best position so
far.
"""
mutable struct Particle
    pos
    velocity
    p_best
    function Particle(pos)
        z = zeros(length(pos))
        new(pos, z, pos)
    end
end


"""
At each step, a particle updates its velocity using the following equation

Afterward, the position is updated using a sigmoid activation function on the
velocity vector.
"""
function move!(p, best, p_acc, acc, mom, r1, r2, r3)
    check_eq(
        "Incongruent vector dimentions",
        length(p.pos), length(best),
        length(r1), length(r2)
    )
    p_best_dir = p_acc * r1 .* (p.p_best - p.pos)
    best_dir = acc * r2 .* (best - p.pos)
    p.velocity = mom * p.velocity + p_best_dir + best_dir
    p.pos = activation.(p.velocity, r3)
end

move!(p, best, p_acc, acc, mom) = move!(
    p, best, p_acc, acc, mom,
    rand(length(best)),
    rand(length(best)),
    rand(length(best))
)

"Sigmoid activation"
activation(v, r) = if r < 1 / (1 + ℯ^v) 1 else 0 end

"""
A swarm of particles. It also keeps track of the current global best and common
particle values, like momentum and acceleration.
"""
mutable struct Swarm
    particles
    best_pos
    acc
    mom
    obj
    function Swarm(num_particles, dim, cost, acc, mom)
        pos = [ random(dim) for i in range(1, num_particles, step=1) ]
        imin = argmin(cost.(pos))
        best_pos = pos[imin]
        particles = Particle.(pos)
        new(particles, best_pos, acc, mom, cost)
    end
end

"""
Moves all particles, updates invidivual bests and global best
"""
function step!(s::Swarm)
    for p in s.particles
        move!(p, s.best_pos, s.acc, s.acc, s.mom)

        if obj(p.pos) < obj(p.p_best)
            p.p_best = p.pos
        end
    end

    for p in s.particles
        if obj(p.pos) < obj(s.best_pos)
            s.best_pos = p.pos
        end
    end
end

end
