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
	using Graphs, GraphRecipes, SimpleWeightedGraphs
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
"""

# ╔═╡ 37ebb301-dfaf-4c04-97fe-71d73a24c109
md"""
Los detalles del movimiento se pueden entender con las siguientes ecuaciones. Si se
tiene a la partícula ``x_{i}`` que a tiempo ``t`` tiene velocidad ``v^{t}_{i}``
entonces la posición en el tiempo ``t+1`` está dada por

```math
\begin{align*}
	v^{t+1}_{i} &=
	v^{t}_{i} + \alpha \epsilon_{1}(x_{i}-\textbf{x}_{i}) 
	+ \beta \epsilon_{2}(x_{i} - \textbf{g})\\
	x^{t+1}_{i} &= x^{t}_{i} + v^{t+1}_{i}
\end{align*}
```

Donde ``\epsilon_{1}, \epsilon_{2}`` son vectores aleatorios con entradas en el
rango ``[0, 1]``, ``\alpha, \beta`` son parámetros globales para determinar que tan
aleatorio será el movimiento de las partículas y ``\textbf{x}_{i}, \textbf{g}`` son 
las mejores posiciones encontradas de ``x_{i}`` y globalmente.
"""

# ╔═╡ 98e190c7-9ce8-460e-939e-1a7d2bf4e85e
md"""
## Optimización Acelerada

En una forma simplificada de la heurística, conocidad como Optimización Acelerada por Enjambre de Partículas (APSO), se sustiyuye la mejor posición histórica por un vector sacado de una distribución normal ``N(0, 1)``. Resulta ser que estadísticamente son suficientemente similares para no afectar la conversión del algoritmo, simplificando la implementación.

"""

# ╔═╡ 0bd42b90-1fd2-4642-a8c7-9bfd5a5003c5
md"""
Al no tener que usar la mejor posición de la partícula, se pueden simplificar las ecuaciones de la siguiente manera

```math
\begin{align*}
	v^{t+1}_{i} &= v^{t}_{i} + \alpha \epsilon + \beta (x_{i} - \textbf{g})\\
	x^{t+1}_{i} &= x^{t}_{i} + v^{t+1}_{i}
\end{align*}
```

con ``\epsilon \in N(0, 1)``
"""

# ╔═╡ caf8bcbd-84db-477a-ae1a-5d76c4e14a84
md"""
Otra simplificación que se puede hacer es eliminar la necesidad de guardar la velocidad y solo considerar la posición anterior junto con la aceleración inmediata. Es decir

```math
x^{t+1}_{i} = x^{t}_{i} +\alpha \epsilon + \beta (x_{i} - g)
= (1 - \beta)x^{t}_{i} +\alpha \epsilon + \beta \textbf{g}
```

Increíblemente, a pesar de todas estas simplificaciones, *APSO* mantiene la propiedad de conversión global, con las ventajas de ser más sencilla de implementar.
"""

# ╔═╡ b6ea2563-e197-4c60-b3f3-8dd4e31aed41
md"""
### Ejemplo

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
	swarm = Swarm(2, f, type = ParticleSwarm.apso, num_particles = 20, range = 0:0.01:6)

	anim = @animate for i in 1:10
		step!(swarm)
		x = [p[1] for p in eachcol(swarm.particles.pos)]
		y = [p[2] for p in eachcol(swarm.particles.pos)]
		contour(xk, xk, (x, y) -> f([x, y]))
		scatter!(x, y, label="")
	end
	
	gif(anim, "/tmp/anim_fps1.gif", fps = 1)
end

# ╔═╡ 37c0d3ff-f84b-43c3-8194-e096acaa422a
md"""
Al final de la simulación, se tiene la siguiente configuración. La mejor posición
fue encontrada en $(swarm.best[1]), con valor de $(swarm.best[2]).
"""

# ╔═╡ 4428ccff-ff9d-47b1-a1f9-d206b5018d0f
begin
	x = [p[1] for p in eachcol(swarm.particles.pos)]
	y = [p[2] for p in eachcol(swarm.particles.pos)]
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


# ╔═╡ efc87d14-8ce6-4061-99ea-f9cdb19c60e2
md"""
## Optimización binaria

Muchos problemas interesantes son de naturaleza combinatoria. Sin embargo, el concepto de velocidad requerido en PSO hace que no haya una manera directa de adaptar la heurística para esos casos.

Por este motivo, los autores originales de la heurística diseñaron una alternativa que funciona sobre espacios binarios, denominada Optiminización por Enjambre de Partículas Binarias.

Como su nombre lo indica, la partículas solo pueden tomar posiciones en ``\mathbb{B}^{n}``. La velocidad sigue siendo un vector en ``\mathbb{R}^{n}``, pero en lugar de afectar directamente la posición, es interpretada como una especie de probabilidad que determina la posición siguiente.
"""

# ╔═╡ 8ebedcdd-8063-4c0a-9200-b3c71c2b458d
md"""
La velocidad no está acotada entre ``[0, 1]``, por lo que se necesita una función que mapée valores de ``\mathbb{R}`` a ``[0, 1]``. Una opción común es la función logística 

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

Con esta, una velocidad ``v^{t}``, la posición ``x^{t+1}`` se puede definir como

```math
x^{t+1}_{i} = \begin{cases} 
  1, r < \sigma(v^{t}_{i}) \\
  0, \text{en otro caso}
\end{cases}
```

``r`` es un número aleatorio entre 0 y 1.
"""

# ╔═╡ fe1dbbf0-0f00-4e56-88ca-196adeba8cda
md"""
Además de actualizar la posición, se debe actualizar la velocidad usando la posición binaria. Para esto se puede construir una tabla analizando como deberían cambiar la velocidad para la particula actual ``v`` se asemeje más a la partícula ``v'``

| ``v_{i}`` | ``v'_{i}`` | ``\Delta v^{t+1}_{i}`` |
| --- | --- | --- |
|  0  |  0  |  0  |
|  0  |  1  |  -  |
|  1  |  0  |  +  |
|  1  |  1  |  0  |

Es decir, basta con tomar ``v_{i} - v'_{i}``, que es precisamente la operación  original realizada para actualizar de la velocidad respecto a las mejores posiciones.

```math
v^{t+1}_{i} = v^{t}_{i} + \alpha \epsilon_{1}(x_{i}-\textbf{x}_{i}) 
	+ \beta \epsilon_{2}(x_{i} - \textbf{g})
```
"""

# ╔═╡ b8036bf9-eb78-45a8-8783-259f09b7d963
md"""
### Ejemplo

Intentemos encontrar la subgráfica generadora mínima de una gráfica completa (que debería ser un árbol).
"""

# ╔═╡ 61bfc818-3a1c-48a9-8663-8d3156194d9f
begin
	n = 5
	g = SimpleWeightedGraph(n)
	for i = 1:n, j = (i+1):n
		add_edge!(g, i, j, (i*j)/10)
	end
end

# ╔═╡ 183caa56-357d-4a37-a699-962ce6a99c68
graphplot(g, edgelabel = g.weights, curves = false)

# ╔═╡ 797a7bb8-0116-4108-ad64-270d4fe95c67
md"""
Para codificar las gráficas generadoras como vectores booleanos, se van a enumerar las aristas, de 1 hasta $(trunc(Int, n*(n-1)/2)). El número de cada arista representará su entrada en el vector booleano. Esa entrada es 1 si y sólo si la arista está presente en la subgráfica.
"""

# ╔═╡ 91f2cf77-8507-4e38-908e-930dae3483a4
function ith_edge(i)
	n = ceil((1 + √(1 + 8 * i)) / 2)
	offset = i - ((n - 1) * (n - 2))/2
	Edge(Int(n), Int(offset))
end

# ╔═╡ bfe74a53-d817-4e40-9270-6b768f9c8f19
function decode_subgraph(g, choices)
	edges = [ith_edge(i) for (i, ok) in enumerate(choices) if ok]
	gs, _ = induced_subgraph(g, edges)
	gs
end

# ╔═╡ a2a08ba3-1677-4060-b022-82982b038cd1
function total_weight(g)
	sum(g.weights)/2
end

# ╔═╡ 1154b1d8-7e66-4c31-8dbd-ab71b54866e7
md"""
El costo base sería solo la suma de los costos de las aristas presentes en la subgráfica. Pero no todas las subgráficas son generadoras.

Para dar preferencia a las que las generadoras, se penaliza a todas las subgráficas que no lo sean.
"""

# ╔═╡ 50d9ec6e-ccd6-49fb-9360-b682179b8fab
function penalized_cost(g, x)
	h = decode_subgraph(g, x)
	penalty = nv(h) < nv(g) ? (nv(g) - nv(h)) : 0
	return total_weight(h) + penalty
end

# ╔═╡ b8dd3676-9a81-42b8-899c-2df7da55d9d4
begin
	num_edges = Int((n * (n - 1)) / 2)
	b_swarm = Swarm(num_edges, x -> penalized_cost(g, x), type = ParticleSwarm.bpso, num_particles = 20)

	for i in 1:50
		step!(b_swarm)
	end
	
	best_subgraph = decode_subgraph(g, b_swarm.best[1])
end

# ╔═╡ e43fb917-947f-4b4c-9f73-862cee20d305
graphplot(best_subgraph, edgelabel = g.weights, curves = false)

# ╔═╡ f12d272d-e51b-41ae-ba3a-deec2774a579
md"""
La mejor gráfica encontrada, con un peso total de $(total_weight(best_subgraph)).
"""

# ╔═╡ Cell order:
# ╠═81af82f5-d396-49f4-83ff-934a8dee0838
# ╠═dbdac828-ce7c-497d-a7e6-16a50e63b8ec
# ╠═b905daf4-f8a3-4f69-ad2c-ac0961899881
# ╟─7370651c-bf0c-11eb-024d-7526a5c8ed91
# ╟─8107a2fb-c903-47bd-a8f2-3ef5efe90bcd
# ╟─37ebb301-dfaf-4c04-97fe-71d73a24c109
# ╟─98e190c7-9ce8-460e-939e-1a7d2bf4e85e
# ╟─0bd42b90-1fd2-4642-a8c7-9bfd5a5003c5
# ╟─caf8bcbd-84db-477a-ae1a-5d76c4e14a84
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
# ╟─efc87d14-8ce6-4061-99ea-f9cdb19c60e2
# ╟─8ebedcdd-8063-4c0a-9200-b3c71c2b458d
# ╟─fe1dbbf0-0f00-4e56-88ca-196adeba8cda
# ╟─b8036bf9-eb78-45a8-8783-259f09b7d963
# ╠═61bfc818-3a1c-48a9-8663-8d3156194d9f
# ╠═183caa56-357d-4a37-a699-962ce6a99c68
# ╟─797a7bb8-0116-4108-ad64-270d4fe95c67
# ╠═91f2cf77-8507-4e38-908e-930dae3483a4
# ╠═bfe74a53-d817-4e40-9270-6b768f9c8f19
# ╠═a2a08ba3-1677-4060-b022-82982b038cd1
# ╟─1154b1d8-7e66-4c31-8dbd-ab71b54866e7
# ╠═50d9ec6e-ccd6-49fb-9360-b682179b8fab
# ╠═b8dd3676-9a81-42b8-899c-2df7da55d9d4
# ╠═e43fb917-947f-4b4c-9f73-862cee20d305
# ╟─f12d272d-e51b-41ae-ba3a-deec2774a579
