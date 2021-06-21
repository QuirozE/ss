### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 81af82f5-d396-49f4-83ff-934a8dee0838
using DrWatson

# ╔═╡ b905daf4-f8a3-4f69-ad2c-ac0961899881
begin
	using Random
	using Plots; plotlyjs();
	using BenchmarkTools
end

# ╔═╡ dbdac828-ce7c-497d-a7e6-16a50e63b8ec
@quickactivate("ss")

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

# ╔═╡ 01c379bb-a179-4cf6-b5e0-e307f1bc7712
begin
	mutable struct Particle{T<:AbstractFloat}
		pos::Array{T, 1}
		vel::Array{T, 1}
		p_best::Array{T, 1}
	end
	
	function Particle(init)
		float_init = convert.(AbstractFloat, init)
		type_init = eltype(float_init)
		Particle(
			float_init,
			zeros(type_init, length(float_init)),
			float_init
		)
	end
end

# ╔═╡ 284a3564-8004-4197-8757-7cb35dfdffa5
p = Particle([1, 0])

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

# ╔═╡ a1e23da6-4dcf-4e16-8a9c-f280eecfb22b
function move!(p, α, β, ϵ1, ϵ2, g)
	p.vel = p.vel .+ α * (ϵ1 .* (p.pos .- g)) .+ β * (ϵ2 .* (p.pos .- p.p_best))
	p.pos = p.pos .+ p.vel
end

# ╔═╡ a11aa894-f58d-49a0-960c-c196dafebbef
begin
	move!(p, 1, 1, [1, 1], [1, 1], [1, 1])
	p
end

# ╔═╡ 8bfee0a5-de38-46ec-8715-c52535033d75
md"""
En la implementación, para evitar variables globales se puede usar un tipo compuesto
que representa al enjambre de partículas.

Además, para tener una mejor visualización más adelante, se incluyen límites en el
espacio de búsqueda. En una aplicación real esto también podría ser útil si ya se sabe
en que áreas no buscar.
"""

# ╔═╡ 4a7d1be4-ab8a-44c2-8845-3977a2e8dcf4
begin
	mutable struct Swarm{T<:AbstractFloat}
		particles::Array{Particle{T}, 1}
		best_pos::Array{T, 1}
		α::T
		β::T
		dims
		min_lim
		max_lim
		cost
		rng
	end	

	function Swarm(dims, n, axis, cost; α=2, β=2, seed=0)
		if(n < 1)
			error("Swarm of at least 1 is required")
		end

		rng = MersenneTwister(seed)
		min = minimum(axis)
		max = maximum(axis)
		float_axis = range(min, max, step=0.1)

		particles = [
			Particle(rand(rng, float_axis, dims)) for _ in 1:n
		]

		g_best = particles[1]
		for p in particles
			if(cost(p.pos) < cost(g_best.pos))
				g_best = p
			end
		end

		Swarm(
			particles,
			g_best.pos,
			convert(AbstractFloat, α),
			convert(AbstractFloat, β),
			dims, min, max, cost, rng
		)
	end
end


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
s = Swarm(2, 5, -500:500, f)

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

# ╔═╡ 6e26965d-d84a-4eee-aef7-f90bc8ff22f3
function step!(s)
	dims = length(s.particles[1].pos)
	ϵ1 = rand(s.rng, range(0, 1, step=0.01), dims)
	ϵ2 = rand(s.rng, range(0, 1, step=0.01), dims)
	
	for p in s.particles
		move!(p, s.α, s.β, ϵ1, ϵ2, s.best_pos)
		
		# bounding particle
		p.pos = min.(p.pos, s.max_lim)
		p.pos = max.(p.pos, s.min_lim)
	
		if s.cost(p.pos) < s.cost(p.p_best) 
			p.p_best = p.pos
		end
	end
	
	for p in s.particles
		if s.cost(p.pos) < s.cost(s.best_pos)
			s.best_pos = p.pos
		end
	end
end

# ╔═╡ 5f1e3197-b813-42f5-afcd-4f0285a37414
md"""
Usando `Plots.jl` se puede generar una visualización de las partículas durante la
optimización.
"""

# ╔═╡ 83dde1eb-2f6f-4032-bc39-2b8cefab91be
begin
	swarm = Swarm(2, 50, 0:4, f)

	anim = @animate for i in 1:10
		step!(swarm)
		x = [p.pos[1] for p in swarm.particles]
		y = [p.pos[2] for p in swarm.particles]
		contour(xk, xk, (x, y) -> f([x, y]))
		scatter!(x, y, label="")
	end
	
	gif(anim, "/tmp/anim_fps5.gif", fps = 3)
end

# ╔═╡ 37c0d3ff-f84b-43c3-8194-e096acaa422a
md"""
Al final de la simulación, se tiene la siguiente configuración. La mejor posición
fue encontrada en $(swarm.best_pos[1], swarm.best_pos[2]), con valor de
$(f(swarm.best_pos))
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

# ╔═╡ 59672487-3d77-456c-bb67-e1865c7409be
function pso(cost; num_particles, dims, axis, steps, α=2, β=2, seed=0)
	swarm = Swarm(dims, num_particles, axis, cost, α=α, β=β, seed=seed)
	for i in 1:steps
		step!(swarm)
	end
	swarm.best_pos
end

# ╔═╡ 7c3c9f60-c070-4e99-8f06-72c870345413
@benchmark pso(f, num_particles=50, dims=2, axis=xk, steps=100)

# ╔═╡ e2b3dc4c-f771-4d15-9577-528b98680236


# ╔═╡ Cell order:
# ╠═81af82f5-d396-49f4-83ff-934a8dee0838
# ╠═dbdac828-ce7c-497d-a7e6-16a50e63b8ec
# ╠═b905daf4-f8a3-4f69-ad2c-ac0961899881
# ╟─7370651c-bf0c-11eb-024d-7526a5c8ed91
# ╟─8107a2fb-c903-47bd-a8f2-3ef5efe90bcd
# ╠═01c379bb-a179-4cf6-b5e0-e307f1bc7712
# ╠═284a3564-8004-4197-8757-7cb35dfdffa5
# ╟─37ebb301-dfaf-4c04-97fe-71d73a24c109
# ╠═a1e23da6-4dcf-4e16-8a9c-f280eecfb22b
# ╠═a11aa894-f58d-49a0-960c-c196dafebbef
# ╟─8bfee0a5-de38-46ec-8715-c52535033d75
# ╠═4a7d1be4-ab8a-44c2-8845-3977a2e8dcf4
# ╠═a1003d54-d733-455e-9fc2-6f9e9965e9b1
# ╟─b6ea2563-e197-4c60-b3f3-8dd4e31aed41
# ╠═d3d2f6ec-4907-4fa7-96c7-a21fd9e13f1f
# ╠═3edce3a8-fc84-43ca-8d5f-383ec45387a8
# ╟─3188e707-fc77-40cb-acbb-2ac92cd5641b
# ╠═6e26965d-d84a-4eee-aef7-f90bc8ff22f3
# ╟─5f1e3197-b813-42f5-afcd-4f0285a37414
# ╠═83dde1eb-2f6f-4032-bc39-2b8cefab91be
# ╟─37c0d3ff-f84b-43c3-8194-e096acaa422a
# ╠═4428ccff-ff9d-47b1-a1f9-d206b5018d0f
# ╟─7be83d4d-71ec-4efc-8368-544d39ccd3e1
# ╠═59672487-3d77-456c-bb67-e1865c7409be
# ╠═7c3c9f60-c070-4e99-8f06-72c870345413
# ╟─e2b3dc4c-f771-4d15-9577-528b98680236
