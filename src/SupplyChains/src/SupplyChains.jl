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

struct SupplyChain
    capacities
    costs

    function SupplyChain(cap, cost)
        check_eq("Incongruent cost-capacity matrices", size(cap), size(cost))
        new(cap, cost)
    end
end

function Base.:(==)(s1::SupplyChain, s2::SupplyChain)
    isequal(s1.capacities, s2.capacities) && isequal(s1.costs, s2.costs)
end

Base.size(chain::SupplyChain) = size(chain.capacities)

struct Schedule
    chain
    max_plants
    max_distributors
    active_plants
    active_distributors
    flow
    function Schedule(chain, mp, md, ap, ad, flow)
        dims = size(chain)
        check_eq("Incongruent flow-chain dims", dims, size(flow))
        check_eq("Incongruent number of plants", length(ap), dims[2])
        check_eq("Incongruent number of distributors", length(ap), dims[3])
        new(chain, mp, md, ap, ad, flow)
    end
end

function Schedule(cap, cost, mp, md, ad, dp, flow)
    chain = SupplyChain(cap, cost)
    Schedule(chain, mp, md, ad, dp, flow)
end

function Schedule(cap, cost, mp, md, ap, ad, flowsp, flowpd, flowdc)
    flow = Flow(flowsp, flowpd, flowdc)
    Schedule(cap, cost, mp, md, ap, ad, flow)
end

cost(s::Schedule) = cost(
    s.chain.costs,
    s.active_plants,
    s.active_distributors,
    s.flow
)

function cost(c, ap, ad, load)
    au_cost = vec(active_flow(c.unitary, ap, ad))
    al_cost = transpose(au_cost) * vec(load)
    f_vec = vcat(c.plants, c.distributors)
    f_cost = transpose(f_vec) * vcat(
        ap, ad
    )
    al_cost + f_cost
end

function is_valid(s)
    true
end

end # module
