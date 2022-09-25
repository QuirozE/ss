using Test, SupplyChains
import SupplyChains:active_flow

@testset "SupplyChains" begin
    @testset "Creation" begin
        z = hcat(0)
        empty_cap = Capacity([0], [0], [0], [0])
        empty_cost = Cost([0], [0], z, z, z)
        empty_flow = Flow(z, z, z)
        empty_chain = SupplyChain(empty_cap, empty_cost, 0, 0)
        @test empty_cap.clients == [0]
        @test empty_cost.plants == [0]
        @test empty_cost.unitary.supls_plants == z
        @test empty_chain.max_plants == 0
    end

    dims = rand(1:10, 4)
    o = [ones(dims[2]), ones(dims[3])]
    om = [
        ones(dims[1], dims[2]), ones(dims[2], dims[3]),
        ones(dims[3], dims[4])
    ]
    zm = [
        zeros(dims[1], dims[2]), zeros(dims[2], dims[3]),
        zeros(dims[3], dims[4])
    ]
    z = [zeros(dims[2]), zeros(dims[3])]
    @testset "Flow" begin
        full_flow = Flow(om[1], om[2], om[3])
        @test active_flow(full_flow, o[1], o[2]) == full_flow

        empty_flow = Flow(zm[1], zm[2], zm[3])
        @test active_flow(full_flow, z[1], z[2]) == empty_flow

        only_dists = Flow(zm[1], zm[2], om[3])
        @test active_flow(full_flow, z[1], o[2]) == only_dists

        supls_plants = zeros(dims[1], dims[2])
        supls_plants[:, 1] = ones(dims[1])
        first_plant = zeros(dims[2])
        first_plant[1] = 1
        first_plant_flow = Flow(
            supls_plants,
            zeros(dims[2], dims[3]),
            zeros(dims[3], dims[4])
        )
        @test active_flow(full_flow, first_plant, z[2]) == first_plant_flow

    end

    @testset "Cost" begin
        full_flow = Flow(om[1], om[2], om[3])
        rm = [
            rand(1:10, (dims[1], dims[2])), rand(1:10, (dims[2], dims[3])),
            rand(1:10, (dims[3], dims[4]))
        ]
        rand_unit_cost = Cost(z[1], z[2], rm[1], rm[2], rm[3])
        @test rand_unit_cost(o[1], o[2], full_flow) == sum(sum.(rm))
        @test rand_unit_cost(z[1], z[2], full_flow) == 0

        sp_only = Flow(om[1], zm[2], zm[3])
        @test rand_unit_cost(o[1], o[2], sp_only) == sum(rm[1])

        rf = [rand(1:10, dims[2]), rand(1:10, dims[3])]
        rand_fixed_cost = Cost(rf[1], rf[2], zm[1], zm[2], zm[3])
        @test rand_fixed_cost(o[1], o[2], full_flow) == sum(sum.(rf))
    end
end
