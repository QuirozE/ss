using Test, SupplyChains

@testset "SupplyChains" begin
    z = hcat(1)
    empty_cap = SupplyCapacity([0], [0], [0], [0])
    empty_cost = SupplyCost([0], [0], z, z, z)
    empty_chain = SupplyChain(empty_cap, empty_cost)
    empty_sch = SupplySchedule(empty_chain, 0, 0, [0], [0], z, z, z)

    @testset "Creation" begin
        @test empty_cap.clients == [0]
        @test empty_cost.plants == [0]
        @test empty_cost.unitary.supls_plants == z
        @test dims(empty_chain) == (1, 1, 1, 1)
        @test empty_sch.max_plants == 0
    end

    @testset "Cost" begin
    @test cost(empty_sch) == 0
    end
end
