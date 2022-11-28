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

Se busca cumplir la demanada mientras se minimice el costo de operación.
"""

# ╔═╡ 3866d9d6-1b67-4da0-b8df-4fac9a03671a
md"""
## Optimizando

El problema se puede dividir en dos
- Elegir que plantas de ensamblaje y puntos de distribución hay que abrir.
- Elegir la cantidad de productos a transportar entre cada punto.

Curiosamente, ambas dependen una de la otra, y ambas tienen restricciones entre sí. Fijando la solución a uno se puede analizar el otro más fácilmente.
"""

# ╔═╡ dc36a7d2-ded1-4a47-a352-a8e5482df169
md"""
### Optimizando puntos abiertos

Esto es un problema sobre un espacio binario, por lo que se necesita una técnica de optimización discreta. La opción aquí implementada es la optimización binaria por enjambre de partículas. 
"""

# ╔═╡ 7101217f-5a2f-4c6c-9128-966581117eea
md"""
### Optimizando cargas

Este problema se puede formalar como una optimización lineal. En Julia está el paquete [JuMP](https://jump.dev/), que es un front end para varias docenas de diferentes optimizadores lineales.
"""

# ╔═╡ b4afea92-38e9-4008-b10e-a89784d00c0f
md"""
### Todo junto
"""

# ╔═╡ 09c89e97-c78e-445b-9ce2-0b173591898a
swarm = SupplyChains.swarm_optimizer(chain)

# ╔═╡ 3e830caa-ecee-4303-a26b-f3ce8180272e
SupplyChains.optimize(swarm)

# ╔═╡ 1e013f02-896a-4f24-b661-c1809b3b7b8c
swarm.obj.best_load

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
# ╟─3866d9d6-1b67-4da0-b8df-4fac9a03671a
# ╟─dc36a7d2-ded1-4a47-a352-a8e5482df169
# ╟─7101217f-5a2f-4c6c-9128-966581117eea
# ╟─b4afea92-38e9-4008-b10e-a89784d00c0f
# ╠═09c89e97-c78e-445b-9ce2-0b173591898a
# ╠═3e830caa-ecee-4303-a26b-f3ce8180272e
# ╠═1e013f02-896a-4f24-b661-c1809b3b7b8c
