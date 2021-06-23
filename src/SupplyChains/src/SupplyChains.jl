module SupplyChains

using LinearAlgebra

export SupplyCapacity, SupplyCost, SupplyChain, SupplySchedule,
    size, cost

function check_eq(mess, e1, e2)
    if e1 != e2
        error(mess * ": $e1 ≠ $e2")
    end
end

struct SupplyCapacity
    supliers
    plants
    distributors
    clients
end

function Base.size(cap::SupplyCapacity)
    length.((cap.supliers, cap.plants, cap.distributors, cap.clients))
end

struct SupplyEdges
    supls_plants
    plants_dists
    dists_clients

    function SupplyEdges(cost_sp, cost_pd, cost_dc)
        _, np = size(cost_sp)
        _np, nd = size(cost_pd)
        check_eq("Incongruent matrices size", np, _np)

        _nd, _ = size(cost_dc)
        check_eq("Incongruent matrices size", nd, _nd)

        new(cost_sp, cost_pd, cost_dc)
    end
end

function active_edges(e, active_plants, active_dists)
    plants_matrix = Diagonal(active_plants)
    dists_matrix = Diagonal(active_dists)
    supl_plants_cost = e.supls_plants * plants_matrix
    plants_dist_cost = plants_matrix * e.plants_dists * dists_matrix
    dist_clients_cost = dists_matrix * e.dists_clients
    SupplyEdges(supl_plants_cost, plants_dist_cost, dist_clients_cost)
end

function Base.vec(e::SupplyEdges)
    sp = vec(e.supls_plants)
    pd = vec(e.plants_dists)
    dc = vec(e.dists_clients)
    vcat(sp, pd, dc)
end

function Base.size(e::SupplyEdges)
    (size(e.supls_plants)..., size(e.dists_clients)...)
end

struct SupplyCost
    plants
    distributors
    unitary

    function SupplyCost(fixed_p, fixed_d, cost_sp, cost_pd, cost_dc)
        unitary_cost = SupplyEdges(cost_sp, cost_pd, cost_dc)
        d = size(unitary_cost)
        check_eq(
            "Incongruent fixed-unitary cost size",
            (length(fixed_p), length(fixed_d)),
            (d[2], d[3])
        )

        new(fixed_p, fixed_d, unitary_cost)
    end
end

Base.size(cost::SupplyCost) = size(cost.unitary)

struct SupplyChain
    capacities
    costs

    function SupplyChain(cap, cost)
        check_eq("Incongruent cost-capacity matrices", size(cap), size(cost))
        new(cap, cost)
    end
end

Base.size(chain::SupplyChain) = size(chain.capacities)

struct SupplySchedule
    chain
    max_plants
    max_distributors
    active_plants
    active_distributors
    load

    function SupplySchedule(chain, mp, md, ap, ad, load)
        dims = size(chain)
        check_eq("Incongruent flow-chain dims", dims, size(load))
        check_eq("Incongruent number of plants", length(ap), dims[2])
        check_eq("Incongruent number of distributors", length(ap), dims[3])
        new(chain, mp, md, ap, ad, load)
    end
end

function SupplySchedule(cap, cost, mp, md, ap, ad, flowsp, flowpd, flowdc)
    chain = SupplyChain(cap, cost)
    flow = SupplyEdges(flowsp, flowpd, flowdc)
    SupplySchedule(chain, mp, md, ap, ad, flow)
end

function cost(s)
    active_unit_cost = vec(active_edges(
        s.chain.costs.unitary,
        s.active_plants,
        s.active_distributors
    ))
    active_load_cost = transpose(active_unit_cost) * vec(s.load)
    fixed_cost = vcat(s.chain.costs.plants, s.chain.costs.distributors)
    println(s.active_plants)
    active_fixed_cost = transpose(fixed_cost) * vcat(
        s.active_plants, s.active_distributors
    )
    active_load_cost + active_fixed_cost
end

end # module
