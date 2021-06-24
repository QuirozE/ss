module SupplyChains

using LinearAlgebra

export Capacity, Flow, Cost, SupplyChain, Schedule,
    size, cost

function check_eq(mess, e1, e2)
    if e1 != e2
        error(mess * ": $e1 ≠ $e2")
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
    vcat(sp, pd, dc)
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
    flow

    function SupplyChain(cap, cost, mp, md, flow)
        check_eq("Incongruent cost-capacity matrices", size(cap), size(cost))
        check_eq("Incongruent chain-flow dims", size(flow), size(cost.unitary))
        new(cap, cost, mp, md, flow)
    end
end

function SupplyChain(cap, cost, mp, md, flowsp, flowpd, flowdc)
    flow = Flow(flowsp, flowpd, flowdc)
    SupplyChain(cap, cost, mp, md, flow)
end

function Base.:(==)(s1::SupplyChain, s2::SupplyChain)
    (
        isequal(s1.capacities, s2.capacities) && isequal(s1.costs, s2.costs)
        && isequal(s1.max_plants, s2.max_plants)
        && isequal(s1.max_distributors, s2.max_distributors)
        && isequal(s1.flow, s2.flow)
    )
end

Base.size(chain::SupplyChain) = size(chain.capacities)

costs(s::SupplyChain, ap, ad) = cost(
    s.costs,
    ap,
    ad,
    s.flow
)

mutable struct Schedule
    chain
    active_plants
    active_distributors
    function Schedule(c, ap, ad)
        dims = size(chain)
        check_eq("Incongruent number of plants", length(ap), dims[2])
        check_eq("Incongruent number of distributors", length(ad), dims[3])
        new(c, ap, ad)
    end
end

cost(s::Schedule) = cost(
    s.chain.costs, s.active_plants,
    s.active_distributors, s.chain.load
)

function update!(s::Schedule, active_vec)
    l = length(active_vec)
    check_eq(
        "Incongruent active nodes vector",
        l,
        length(s.active_plants) + length(s.active_distributors)
    )

    p = length(s.active_plants)

    s.active_plants = active_vec[1:p]
    s.active_distributors = active_vec[p+1:l]
end

function is_valid(s)
    true
end

module PSO

mutable struct Particle
    velocity
    pos
    p_best
    schedule
end

function move!()
end

end

end # module
