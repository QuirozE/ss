using Test, SupplyChains

z = hcat(1)

empty_cost = SupplyCost([0, 0], z, z, z)

empty_chain = SupplyChain((1, 1, 1, 1), 1, 1, [0, 0, 0, 0], empty_cost)

@testset "SupplyChains Creation" begin
    @test empty_cost.fixed_cost == [0, 0]
    @test empty_cost.unitary_cost[1][1] == 1
    @test empty_chain.num_suppliers == 1
end
