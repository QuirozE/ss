### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 81af82f5-d396-49f4-83ff-934a8dee0838
using DrWatson

# ╔═╡ dbdac828-ce7c-497d-a7e6-16a50e63b8ec
quickactivate("..", "ss")

# ╔═╡ b905daf4-f8a3-4f69-ad2c-ac0961899881
begin
	using Random
	using Plots
	using BenchmarkTools
	using ParticleSwarm
end

# ╔═╡ 7370651c-bf0c-11eb-024d-7526a5c8ed91
md"""
# Optimización por ejambre de partículas

La optimización por enjambre de partículas (_PSO_ por sus siglas en inglés) es una
técnica de optimización para funciones contínuas fue desarrollada por Kennedy y
Eberhart en 1995. Es parte de una familia más extensa de algoritmos de optimización
llamados _Inteligencia de enjambre_.
"""

# ╔═╡ 8107a2fb-c903-47bd-a8f2-3ef5efe90bcd
md"""
Este algoritmo intenta imitar los patrones observados en parvadas de aves y
cardumenes de peces. Abstrae estos agentes individuales como partículas que se mueven
a través de un espacio de búsqueda.

En cada ronda, cada particula intenta moverse en la dirección de la mejor posición que
hayan encontrado y hacia la dirección de la mejor posición global encontrada. También
se introduce cierta aleatoriedad para fomentar la exploración de nuevas áreas.

En una forma simplificada de la heurística, conocidad como Optimización Acelerada por Enjambre de Partículas (APSO), se sustiyuye la mejor posición histórica por un vector sacado de una distribución normal ``N(0, 1)``. Resulta ser que estadísticamente son suficientemente similares para no afectar la conversión del algoritmo, simplificando la implementación.
"""

# ╔═╡ 284a3564-8004-4197-8757-7cb35dfdffa5
p = AccParticle([1.0, 0.5])

# ╔═╡ 37ebb301-dfaf-4c04-97fe-71d73a24c109
md"""
Los detalles del movimiento se pueden entender con las siguientes ecuaciones. Si se
tiene a la partícula ``x_{i}`` que a tiempo ``t`` tiene velocidad ``v^{t}_{i}``
entonces la posición en el tiempo ``t+1`` está dada por

```math
\begin{align*}
	v^{t+1}_{i} &=
	v^{t}_{i} + \alpha \epsilon_{1}(x_{i}-\textbf{g}) 
	+ \beta \epsilon_{2}(x_{i} - \textbf{x}_{i})\\
	x^{t+1}_{i} &= x^{t}_{i} + v^{t+1}_{i}
\end{align*}
```

Donde ``\epsilon_{1}, \epsilon_{2}`` son vectores aleatorios con entradas en el
rango ``[0, 1]``, ``\alpha, \beta`` son parámetros globales para determinar que tan
aleatorio será el movimiento de las partículas y ``\textbf{x}_{i}, \textbf{g}`` son 
las mejores posiciones encontradas de ``x_{i}`` y globalmente.
"""

# ╔═╡ a11aa894-f58d-49a0-960c-c196dafebbef
ParticleSwarm.move!(p, p, 0.5, 0.5)

# ╔═╡ 8bfee0a5-de38-46ec-8715-c52535033d75
md"""
En la implementación, para evitar variables globales se puede usar un tipo compuesto
que representa al enjambre de partículas.

Además, para tener una mejor visualización más adelante, se incluyen límites en el
espacio de búsqueda. En una aplicación real esto también podría ser útil si ya se sabe
en que áreas no buscar.
"""

# ╔═╡ b6ea2563-e197-4c60-b3f3-8dd4e31aed41
md"""
Con fines ilustrativos, se intentará minimizar la [función de Michelewicz]
(https://www.sfu.ca/~ssurjano/michal.html)

```math
f(x_{1}, \dots, x_{d}) = - \sum_{i=1}^{d}{
	\sin{x_{i}}
	\sin^{2m}(\frac{ix_{i}^{2}}{\pi})
}
```

con ``d=2`` y ``m=10``, que son las configuraciones más sencillas para visualizar
la función.
"""

# ╔═╡ d3d2f6ec-4907-4fa7-96c7-a21fd9e13f1f
f(x; m = 10) = - sin(x[1])*(sin(x[1]^2/π)^(2*m)) - sin(x[2])*(sin(2*x[2]^2/π)^(2m))

# ╔═╡ a1003d54-d733-455e-9fc2-6f9e9965e9b1
s = ParticleSwarm.Swarm(2, f, num_particles = 5)

# ╔═╡ 3edce3a8-fc84-43ca-8d5f-383ec45387a8
begin
	xk = range(0, 4, step=0.1)
	surface(xk, xk, (x, y) -> f([x, y]))
end

# ╔═╡ 3188e707-fc77-40cb-acbb-2ac92cd5641b
md"""
Entonces, en cada paso, se mueve cada partícula y se actualiza su mejor posición
encontrada usando la función de costo, teniendo en cuenta los límites del espacio de
búsqueda. Al terminar eso, se actualiza la mejor posición encontrada globalmente.
"""

# ╔═╡ 5f1e3197-b813-42f5-afcd-4f0285a37414
md"""
Usando `Plots.jl` se puede generar una visualización de las partículas durante la
optimización.
"""

# ╔═╡ 83dde1eb-2f6f-4032-bc39-2b8cefab91be
begin
	swarm = Swarm(2, f, num_particles = 20, range = 0:0.01:4)

	anim = @animate for i in 1:10
		step!(swarm)
		x = [p.pos[1] for p in swarm.particles]
		y = [p.pos[2] for p in swarm.particles]
		contour(xk, xk, (x, y) -> f([x, y]))
		scatter!(x, y, label="")
	end
	
	gif(anim, "/tmp/anim_fps5.gif", fps = 1)
end

# ╔═╡ 37c0d3ff-f84b-43c3-8194-e096acaa422a
md"""
Al final de la simulación, se tiene la siguiente configuración. La mejor posición
fue encontrada en $(swarm.best_particle[1], swarm.best_particle[2]), con valor de
$(f(swarm.best_particle))
"""

# ╔═╡ 4428ccff-ff9d-47b1-a1f9-d206b5018d0f
begin
	x = [p.pos[1] for p in swarm.particles]
	y = [p.pos[2] for p in swarm.particles]
	contour(xk, xk, (x, y) -> f([x, y]))
	scatter!(x, y, label="")
end

# ╔═╡ 7be83d4d-71ec-4efc-8368-544d39ccd3e1
md"""
¿Y qué tan eficiente es este método? Habría que hacer un _benchmark_. Para esto, se
tiene que colocar todo el código anterior en una función.
"""

# ╔═╡ 7c3c9f60-c070-4e99-8f06-72c870345413
@benchmark pso(f, 2, steps=1000)

# ╔═╡ e2b3dc4c-f771-4d15-9577-528b98680236


# ╔═╡ Cell order:
# ╠═81af82f5-d396-49f4-83ff-934a8dee0838
# ╠═dbdac828-ce7c-497d-a7e6-16a50e63b8ec
# ╠═b905daf4-f8a3-4f69-ad2c-ac0961899881
# ╟─7370651c-bf0c-11eb-024d-7526a5c8ed91
# ╟─8107a2fb-c903-47bd-a8f2-3ef5efe90bcd
# ╠═284a3564-8004-4197-8757-7cb35dfdffa5
# ╟─37ebb301-dfaf-4c04-97fe-71d73a24c109
# ╠═a11aa894-f58d-49a0-960c-c196dafebbef
# ╟─8bfee0a5-de38-46ec-8715-c52535033d75
# ╠═a1003d54-d733-455e-9fc2-6f9e9965e9b1
# ╟─b6ea2563-e197-4c60-b3f3-8dd4e31aed41
# ╠═d3d2f6ec-4907-4fa7-96c7-a21fd9e13f1f
# ╠═3edce3a8-fc84-43ca-8d5f-383ec45387a8
# ╟─3188e707-fc77-40cb-acbb-2ac92cd5641b
# ╟─5f1e3197-b813-42f5-afcd-4f0285a37414
# ╠═83dde1eb-2f6f-4032-bc39-2b8cefab91be
# ╟─37c0d3ff-f84b-43c3-8194-e096acaa422a
# ╠═4428ccff-ff9d-47b1-a1f9-d206b5018d0f
# ╟─7be83d4d-71ec-4efc-8368-544d39ccd3e1
# ╠═7c3c9f60-c070-4e99-8f06-72c870345413
# ╟─e2b3dc4c-f771-4d15-9577-528b98680236
