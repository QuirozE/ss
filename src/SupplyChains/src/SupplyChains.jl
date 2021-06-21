module SupplyChains

export SupplyCapacity, SupplyCost, SupplyChain, SupplySchedule,
    dims, cost

function check_eq(mess, e1, e2)
    if (e1 != e2)
        error(mess * ": $e1 ≠ $e2")
    end
end

struct SupplyCapacity
    supliers
    plants
    distributors
    clients
end

function dims(cap::SupplyCapacity)
    length.((cap.supliers, cap.plants, cap.distributors, cap.clients))
end

struct SupplyEdges
    supls_plants
    plants_dist
    dists_clients

    function SupplyEdges(cost_sp, cost_pd, cost_dc)
        _, np = size(cost_sp)
        _np, nd = size(cost_pd)
        check_eq("Incongruent unitary cost dims", np, _np)

        _nd, _ = size(cost_dc)
        check_eq("Incongruent unitary cost dims", nd, _nd)

        new(cost_sp, cost_pd, cost_dc)
    end
end

function dims(e::SupplyEdges)
    (size(e.supls_plants)..., size(e.dists_clients)...)
end

struct SupplyCost
    plants
    distributors
    unitary

    function SupplyCost(fixed_p, fixed_d, cost_sp, cost_pd, cost_dc)
        unitary_cost = SupplyEdges(cost_sp, cost_pd, cost_dc)
        d = dims(unitary_cost)
        check_eq(
            "Incongruent fixed-unitary cost dims",
            (length(fixed_p), length(fixed_d)),
            (d[2], d[3])
        )

        new(fixed_p, fixed_d, unitary_cost)
    end
end

dims(cost::SupplyCost) = dims(cost.unitary)

struct SupplyChain
    capacity
    cost

    function SupplyChain(cap, cost)
        check_eq("Incongruent cost-capacity dims", dims(cap), dims(cost))
        new(cap, cost)
    end
end

dims(chain::SupplyChain) = dims(chain.capacity)

struct SupplySchedule
    chain
    max_plants
    max_distributors
    active_plants
    active_distributors
    load

    function SupplySchedule(chain, mp, md, ap, ad, flow_sp, flow_pd, flow_dc)
        flow = SupplyEdges(flow_sp, flow_pd, flow_dc)
        new(chain, mp, md, ap, ad, flow)
    end
end

function cost(s::SupplySchedule)
    total_active = transpose(vcat(s.active_plants, s.active_distributors))
    fixed_cost = vcat(s.chain.cost.plants, s.chain.cost.distributors)
    total_active * fixed_cost
end

end # module
