using Test, SupplyChains

@testset "SupplyChains" begin
    z = hcat(0)
    empty_cap = SupplyCapacity([0], [0], [0], [0])
    empty_cost = SupplyCost([0], [0], z, z, z)
    empty_sch = SupplySchedule(empty_cap, empty_cost, 0, 0, [0], [0], z, z, z)

    @testset "Creation" begin
        @test empty_cap.clients == [0]
        @test empty_cost.plants == [0]
        @test empty_cost.unitary.supls_plants == z
        @test empty_sch.max_plants == 0
    end

    @testset "Cost" begin
        o = [1 1; 1 1]
        z2 = [0, 0]
        r = rand(1:10, (2, 2))
        rand_unit_cost_schedule = SupplySchedule(
            SupplyCapacity(z2, z2, z2, z2),
            SupplyCost(z2, z2, r, r, r),
            0, 0, [1, 1], [1, 1], o, o, o
        )
        dims = size(rand_unit_cost_schedule.chain)
        n_edges = dims[1]*dims[2]+dims[2]*dims[3]+dims[3]*dims[4]
        @test cost(rand_unit_cost_schedule) == 12*sum(r)
    end
end
