module SupplyChains

export SupplyCost, SupplyChain

struct SupplyCost
    fixed_cost
    unitary_cost

    function SupplyCost(fixed_cost, cost_sp, cost_pd, cost_dc)
        _, np = size(cost_sp)
        _np, nd = size(cost_pd)

        if np != _np
            error("Incongruent cost matrices length")
        end

        _nd, _ = size(cost_dc)

        if nd != _nd
            error("Incongruent cost matrices length")
        end

        if length(fixed_cost) != np + nd
            error("Incongruent fixed cost vector and unitary cost matrices")
        end

        new(fixed_cost, [cost_sp, cost_pd, cost_dc])
    end
end

struct SupplyChain
    num_suppliers
    num_plants
    num_dist
    num_clients
    max_plants
    max_dist
    capacity
    supply_cost

    function SupplyChain(dims, mp, md, cap, cost)
        total_nodes = sum(dims)

        if length(cap) != total_nodes
            error("Invalid capacity vector")
        end

        new(dims[1], dims[2], dims[3], dims[4], mp, md, cap, cost)
    end
end

end # module
