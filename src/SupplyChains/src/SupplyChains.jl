module SupplyChains

using LinearAlgebra

export Capacity, Flow, Cost, SupplyChain,
    size, cost, length, split_pos, is_valid

function check_eq(mess, e0, es...)
    for (i, e) in enumerate(es)
        if e0 != e
            error(mess * "at element $i: $e0 ≠ $e")
        end
    end
end

struct Capacity
    suppliers
    plants
    distributors
    clients
end

function Base.:(==)(c1::Capacity, c2::Capacity)
    (
        isequal(c1.suppliers, c2.suppliers) &&
        isequal(c1.plants, c2.plants) &&
        isequal(c1.distributors, c2.distributors) &&
        isequal(c1.clients, c2.clients)
    )
end

function Base.size(cap::Capacity)
    length.((cap.suppliers, cap.plants, cap.distributors, cap.clients))
end

struct Flow
    supls_plants
    plants_dists
    dists_clients

    function Flow(cost_sp, cost_pd, cost_dc)
        _, np = size(cost_sp)
        _np, nd = size(cost_pd)
        check_eq("Incongruent matrices size", np, _np)

        _nd, _ = size(cost_dc)
        check_eq("Incongruent matrices size", nd, _nd)

        new(cost_sp, cost_pd, cost_dc)
    end
end

function Base.:(==)(f1::Flow, f2::Flow)
    (
        isequal(f1.supls_plants, f2.supls_plants)
        && isequal(f1.plants_dists, f2.plants_dists)
        && isequal(f1.dists_clients, f2.dists_clients)
    )
end

function active_flow(e, active_plants, active_dists)
    plants_matrix = Diagonal(active_plants)
    dists_matrix = Diagonal(active_dists)
    supl_plants_cost = e.supls_plants * plants_matrix
    plants_dist_cost = plants_matrix * e.plants_dists * dists_matrix
    dist_clients_cost = dists_matrix * e.dists_clients
    Flow(supl_plants_cost, plants_dist_cost, dist_clients_cost)
end

function Base.vec(e::Flow)
    sp = vec(e.supls_plants)
    pd = vec(e.plants_dists)
    dc = vec(e.dists_clients)
    [sp; pd; dc]
end

function Base.size(e::Flow)
    (size(e.supls_plants)..., size(e.dists_clients)...)
end

struct Cost
    plants
    distributors
    unitary

    function Cost(fixed_p, fixed_d, cost_sp, cost_pd, cost_dc)
        unitary_cost = Flow(cost_sp, cost_pd, cost_dc)
        d = size(unitary_cost)
        check_eq(
            "Incongruent fixed-unitary cost size",
            (length(fixed_p), length(fixed_d)),
            (d[2], d[3])
        )

        new(fixed_p, fixed_d, unitary_cost)
    end
end

function Base.:(==)(c1::Cost, c2::Cost)
    (
        isequal(c1.plants, c2.plants)
        && isequal(c1.distributors, c2.distributors)
        && isequal(c1.unitary, c2.unitary)
    )
end

Base.size(cost::Cost) = size(cost.unitary)

function cost(c, ap, ad, load)
    dims = size(c)
    check_eq("Incongruent number of plants", length(ap), dims[2])
    check_eq("Incongruent number of distributors", length(ad), dims[3])

    au_cost = vec(active_flow(c.unitary, ap, ad))
    al_cost = transpose(au_cost) * vec(load)
    f_vec = [c.plants ; c.distributors]
    f_cost = transpose(f_vec) * [ap; ad]
    al_cost + f_cost
end

struct SupplyChain
    capacities
    costs
    max_plants
    max_distributors

    function SupplyChain(cap, cost, mp, md)
        check_eq("Incongruent cost-capacity matrices", size(cap), size(cost))
        new(cap, cost, mp, md)
    end
end

function Base.:(==)(s1::SupplyChain, s2::SupplyChain)
    (
        isequal(s1.capacities, s2.capacities) && isequal(s1.costs, s2.costs)
        && isequal(s1.max_plants, s2.max_plants)
        && isequal(s1.max_distributors, s2.max_distributors)
    )
end

Base.size(chain::SupplyChain) = size(chain.capacities)

cost(s::SupplyChain, ap, ad, flow) = cost(
    s.costs,
    ap,
    ad,
    flow
)

function cost(s::SupplyChain, pos_vec, flow)
    as = split_pos(s, pos_vec)
    cost(s, as[1], as[2], flow)
end

function split_pos(s::SupplyChain, pos_vec)
    l = length(pos_vec)
    dims = size(s)
    check_eq("Incongruent active nodes vector", l, dims[2] + dims[3])
    (pos_vec[1:dims[2]], pos_vec[dims[2] + 1:l])
end

function is_valid(s, pos_vec, flow)
    true
end

module ParticleSwarm

import ..SupplyChains: cost, check_eq

export Particle, Swarm, move!, step!

mutable struct Particle
    pos
    velocity
    momentum
    p_best
    p_acc
    function Particle(pos, m, acc)
        z = zeros(length(pos))
        new(pos, z, m, pos, acc)
    end
end

function move!(p::Particle, best, acc, r1, r2, r3)
    check_eq(
        "Incongruent vector dimentions",
        length(p.pos), length(best),
        length(r1), length(r2)
    )
    p_best_dir = p.p_acc * r1 .* (p.p_best - p.pos)
    best_dir = acc * r2 .* (best - p.pos)
    p.velocity = p.momentum * p.velocity + p_best_dir + best_dir
    p.pos = activation.(p.velocity, r3)
end

move!(p::Particle, best, acc) = move!(
    p, best, acc,
    rand(length(best)),
    rand(length(best)),
    rand(length(best))
)

activation(v, r) = if r < 1 / (1 + ℯ^v) 1 else 0 end

mutable struct Swarm
    particles
    best_pos
    best_acc
    obj
    opt
end

function step!(s::Swarm)
    # Some LP optimization
    extra = s.opt.(s.particles, s.obj)
    # End LP optimization

    for (i, p) in enumerate(s.particles)
        move!(p, s.best_pos, s.best_acc)

        if cost(s.obj, p.pos, extra[i]) < cost(s.obj, p.p_best, extra[i])
            p.p_best = p.pos
        end
    end

    for (i, p) in enumerate(s.particles)
        if cost(s.obj, p.pos, extra[i]) < cost(s.obj, s.best_pos, extra[i])
            s.best_pos = p.pos
        end
    end
end

end

end # module
