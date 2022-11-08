### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ a93399c0-5f25-11ed-3062-05932eb2164c
using DrWatson

# ╔═╡ 5f0a8cec-0da9-4ed5-aa2b-db110eaad363
quickactivate("..", "ss")

# ╔═╡ 6b4dfb3c-3eef-4da8-bea3-2578ed94fbc6
begin
	using SupplyChains
	using Plots
	using Statistics
end

# ╔═╡ 03381c4f-8e7b-4b10-abcb-e245cf997e4c
md"""
# Cadenas de suministros

La logística en un sistema es un mecanismo para satisfacer una demanda, como materiales brutos o tiempo, bajo alguna restricción, normalmente un presupuesto, mientras se optimiza un costo.

En un ambiente industrial, la logística se denomica cadena de suministros. Usualmente tiene dos etapas: una cadena de producción y una cadena de distribucioń.

En este ejemplo, la cadena de producción tiene una cantidad fija de provedores de materias brutas y algunas plantas de ensamblaje. La cadena de distribución tiene centro de distribución y puntos de venta.
"""

# ╔═╡ 833ee11b-e9e6-43a3-bcdc-2034cad5a053
md"""
Cada instalación tiene un costo fijo de operación, y el transporte entre instalaciones tiene un costo por unidad transportada. Provedores, plantas de ensamblaje y centro de distribución tienen un capacidad máxima, y los puntos de venta tiene una demanda.
"""

# ╔═╡ 9e412587-d775-441a-9b8e-d83d6f2085bb
cap = Capacity(
	[500, 650, 390],
	[400, 550, 490, 300, 500],
	[530, 590, 400, 370, 580],
	[460, 330, 450, 300]
)

# ╔═╡ b00d24e1-d266-464c-82a9-03e3cbfa94ab
cost = Cost(
	[1800, 900, 2100, 1100, 900],
	[1000, 900, 1600, 1500, 1400],
	[5 6 4 7 5;
	 6 5 6 6 8;
	 7 6 3 9 6],
	[5 8 4 3 5;
	 8 7 8 6 8;
	 4 7 4 5 4;
	 3 5 3 5 3;
	 5 6 6 8 3],
	[7 4 5 6;
	 5 4 6 7;
	 7 5 3 6;
	 3 5 6 4;
	 4 6 5 7]
)

# ╔═╡ 6c91b460-182c-42c2-969c-05b859fae887
chain = SupplyChain(cap, cost, 4, 4)

# ╔═╡ 8afc4759-3806-43a2-93ee-3db7c1a66d68
md"""
En este ejemplo, la cadena de suministros tiene $(size(chain)[1]) proveedores, $(size(chain)[2]) plantas de ensamblaje, $(size(chain)[3]) plantas de distribuión y $(size(chain)[4]) punto de venta.
"""

# ╔═╡ 16f7c9d8-bda4-4fd0-aec8-22cb9f0a2d2f
md"""
Se busca cumplir la demanada mientras se minimice el costo de operación. Vamos a explorar algunas maneras de aproximar esto.
"""

# ╔═╡ f70d4e66-1537-4a51-aab5-bf1a04d8b48f
md"""
## Medir la calidad de una solución

Para simplificar el problema, primero vamos a fijar un flujo, para solo optimizar cuáles plantas de ensamblaje y centro de distribución hay que abrir. En este caso hay 5+5 = 10 deciciones a tomar, lo cuál resulta en ``2^{10} = 1024`` posibles soluciones. Esto es manejable, por lo que se pueden calcular todas.

Con esto se puede verificar la frecuencia de los costos para determinar la calidad de las soluciones propuestas.
"""

# ╔═╡ 254a8405-ae97-49a2-824e-ab81c82ab372
function all_binaries(n)
	if n < 2
		[[true], [false]]
	else
		bs = all_binaries(n-1)
		vcat([push!(copy(b), true) for b in bs], [push!(copy(b), false) for b in bs])
	end
end

# ╔═╡ 14da7499-9e0e-4901-8ac3-f666a8fd2189
begin
	solutions = all_binaries(10)
	costs = [SupplyChains.particle_cost(chain, s) for s in solutions]
	valid_costs = [c for c in costs if c < 10^5]
	histogram(valid_costs)
end

# ╔═╡ 697b20e8-e0dd-4ab5-aded-764982f69992
dec_q = quantile(valid_costs, 0.0:0.1:1)

# ╔═╡ 43ace981-60a8-4b08-9613-b2895d7eddfc
md"""
## Optimización binaria
"""

# ╔═╡ 09c89e97-c78e-445b-9ce2-0b173591898a
best_particle, best_flow = SupplyChains.optimize(chain, steps = 200)

# ╔═╡ 3e830caa-ecee-4303-a26b-f3ce8180272e
SupplyChains.particle_cost(chain, best_particle)

# ╔═╡ 54d3d36f-8613-420f-8ba0-a42893b8db3c
md"""
## Programación Lineal
"""

# ╔═╡ 26ad1160-341e-4323-a49d-4dfa5d990208
md"""
## Todo junto
"""

# ╔═╡ Cell order:
# ╠═a93399c0-5f25-11ed-3062-05932eb2164c
# ╠═5f0a8cec-0da9-4ed5-aa2b-db110eaad363
# ╠═6b4dfb3c-3eef-4da8-bea3-2578ed94fbc6
# ╟─03381c4f-8e7b-4b10-abcb-e245cf997e4c
# ╟─833ee11b-e9e6-43a3-bcdc-2034cad5a053
# ╠═9e412587-d775-441a-9b8e-d83d6f2085bb
# ╠═b00d24e1-d266-464c-82a9-03e3cbfa94ab
# ╠═6c91b460-182c-42c2-969c-05b859fae887
# ╟─8afc4759-3806-43a2-93ee-3db7c1a66d68
# ╟─16f7c9d8-bda4-4fd0-aec8-22cb9f0a2d2f
# ╟─f70d4e66-1537-4a51-aab5-bf1a04d8b48f
# ╠═254a8405-ae97-49a2-824e-ab81c82ab372
# ╠═14da7499-9e0e-4901-8ac3-f666a8fd2189
# ╠═697b20e8-e0dd-4ab5-aded-764982f69992
# ╟─43ace981-60a8-4b08-9613-b2895d7eddfc
# ╠═09c89e97-c78e-445b-9ce2-0b173591898a
# ╠═3e830caa-ecee-4303-a26b-f3ce8180272e
# ╟─54d3d36f-8613-420f-8ba0-a42893b8db3c
# ╟─26ad1160-341e-4323-a49d-4dfa5d990208
