### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 15e741a0-9c9a-49ce-8057-a47cdbe97817
using DrWatson

# ╔═╡ 84e93c6d-2059-49d1-8d91-39cb8d0a3d15
begin
	using Images
end

# ╔═╡ dd2a9d42-d66c-41c2-9add-67ffe0a6de78
@quickactivate "ss"

# ╔═╡ 21071680-ca2a-11eb-3cf9-613e2bcd45ab
md"""
# Conjuntos de Julia

Dado un número complejo ``z_{0}``, se puede definir la sucesión
```math
z_{n+1} = z^{2}_{n} + c 
```

Si ``\lim_{n \to \infty} {z_n}`` converge, entonces ``z_{0}`` es parte del conjunto de
Julia $J_{c}$ correspondiente a ``c``.
"""

# ╔═╡ 8cc20724-f71f-4079-baa9-648ac6409c6b
suc(z, c) = z*z + c

# ╔═╡ 28ae7b2b-ef1b-424d-905b-78792129414d
md"""
En la implementación, ``n`` se simula con la variable `iters`, que en este caso basta
con `iters=200`. Y para determinar si ``z_{n}`` diverge, simplemente se fija un límite
`threshold=1000`. Si al terminar las iteraciones no se ha superado el límite, se
considera que el número es parte del conjunto.
"""

# ╔═╡ 86efac85-6d1e-42af-aed3-9bb7de1fb69b
function in_julia(z; c=0, iters=200, threshold=1000)
	for _ in 1:iters
		z = suc(z, c)
		if abs(z) > threshold
			return false
		end
	end
	true
end

# ╔═╡ b06181d9-a3b2-4384-8482-dea7ab14c721
function julia_set(c; center=[0, 0], side=3, length=500)
	x_axis = range(
		center[1] - side/2,
		center[1] + side/2,
		length=length
	)
	
	y_axis = range(
		center[2] - side/2,
		center[2] + side/2,
		length=length
	)
	
	[
		in_julia(j+i*im, c=c) for i in x_axis, j in y_axis
	]
end

# ╔═╡ 7188853c-fb3d-49ba-b01b-80d6faf865c0
md"""
Como la función `in_julia` regresa un valor booleano, esta se puede transformar a un
pixel blanco o negro. Así que se puede visualizar el arreglo resultante como una
imagen.
"""

# ╔═╡ 918f516c-e09e-4c0b-9ecd-abdd9febe2b6
md"""
Algunos valores interesantes para `c`.
"""

# ╔═╡ f666fb53-462c-45d2-9432-46896c4d5f32
cs = [
	-0.74543+0.11301im,
	-0.8 + 0.156im,
	-0.7269 + 0.1889im
]

# ╔═╡ 6913d2ba-d577-44b3-a46a-6745fb0fa168
Gray.(julia_set(cs[1], side=3.5, length=3000))

# ╔═╡ Cell order:
# ╠═15e741a0-9c9a-49ce-8057-a47cdbe97817
# ╠═dd2a9d42-d66c-41c2-9add-67ffe0a6de78
# ╠═84e93c6d-2059-49d1-8d91-39cb8d0a3d15
# ╟─21071680-ca2a-11eb-3cf9-613e2bcd45ab
# ╠═8cc20724-f71f-4079-baa9-648ac6409c6b
# ╟─28ae7b2b-ef1b-424d-905b-78792129414d
# ╠═86efac85-6d1e-42af-aed3-9bb7de1fb69b
# ╠═b06181d9-a3b2-4384-8482-dea7ab14c721
# ╟─7188853c-fb3d-49ba-b01b-80d6faf865c0
# ╟─918f516c-e09e-4c0b-9ecd-abdd9febe2b6
# ╠═f666fb53-462c-45d2-9432-46896c4d5f32
# ╠═6913d2ba-d577-44b3-a46a-6745fb0fa168
