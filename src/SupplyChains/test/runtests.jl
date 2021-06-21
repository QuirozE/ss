using Test, SupplyChains

@testset "SupplyChains" begin
    z = hcat(1)
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
    @test cost(empty_sch) == 0
    end
end
