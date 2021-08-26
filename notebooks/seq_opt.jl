### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ fef93ec6-cef7-4d2e-bf3b-2703b16b9d16
using DrWatson

# ╔═╡ fd838fce-ea7f-473d-8e35-ad9c3378c763
using Random

# ╔═╡ 7cac368d-0e6d-40a2-9b26-6c8a3fcbb446
using BenchmarkTools

# ╔═╡ 80e0636e-8642-4ff5-ad88-04be6af2d2bf
using Profile

# ╔═╡ 75c847f1-e41d-4615-9b6e-dac1194785f1
@quickactivate "ss"

# ╔═╡ 074fd91e-989c-11eb-2ee5-51c2e2315347
md"""
# Optimización secuencial

Si se quiere obtener una optimización de recursos (tiempo y memoria) al usar una
solución paralela, primero hay que asegurarse que cada sección secuencial utilice al
máximo los recursos a su disposición.

En Julia hay muchas herramientas y consejos para obtener el mejor desempeño de un
programa secuencial.
"""

# ╔═╡ 0c73012e-1141-4e0f-8b88-6212b65347fb
abstract type Cypher end

# ╔═╡ 34ed55b5-f3cb-4207-8fc7-445e5599a88d
begin
	struct CesarCypher <: Cypher
		key::Int64
		alf::Array{Char}
		pos::Dict{Char, Int64}
	end
	function CesarCypher(key, alf)
		pos = Dict()
		for (i, letter) in enumerate(alf)
			pos[letter] = i
		end
		CesarCypher(key, alf, pos)
	end
end

# ╔═╡ d9ef4c0e-4593-4f10-aeab-8797422f9fdc
function cypher(c::CesarCypher, letter)
	new_index = mod(c.pos[letter] + c.key - 1, length(c.alf)) + 1
	c.alf[new_index]
end

# ╔═╡ 7c49349b-6ed4-4055-8d9d-a4f1488bb46f
begin
	alf = ['A', 'B', 'C', 'D', 'F', 'G']
	key = 4
	cesar = CesarCypher(key, alf)
end

# ╔═╡ 0f52da77-8177-4b1b-8868-5fff10949f30
function cypher_text(c, text)
	s = ""
	for letter in text
		s = string(s, cypher(c, letter))
	end
	s
end

# ╔═╡ 804acee5-3755-450d-9a27-894a2451332f
begin
	small_text = randstring(alf, 10^3)
	big_text = randstring(alf, 10^4)
end

# ╔═╡ 5c9352af-55ee-4031-8c80-2345698c5ee0
md"""
## Medir

No se puede saber si el desempeño de un programa ha mejorado si no se mide su
desempeño actual. Para los dos recusos más importantes (tiempo y memoria) hay
herramientas para medir de forma bastante precisa su desempeño.
"""

# ╔═╡ 0be9c339-82ba-4dda-aa87-275940e0dce5
md"""
Está el módulo `BenchmarkTools.jl` que cuenta con diversas herramientas. La más
sencilla es el macro `@btime`, que simplemente mide el tiempo y memoria usado al
correr una función.
"""

# ╔═╡ bb69abfb-8831-43a4-ab16-dec8d8369667
@show @btime cypher_text(cesar, small_text)

# ╔═╡ 351c228e-ccc7-44e3-9007-1f09f1919daf
md"""
Para un análisis más completo, se puede usar el macro `@benchmark`, que realiza un
análsis estadístico más detallado, corriendo varias veces la función.
"""

# ╔═╡ c305a0b9-8ea9-4f31-ade3-dbdb39ff4686
@benchmark cypher_text(cesar, small_text)

# ╔═╡ a2d08c5d-0572-4720-a37f-1f6ed2a5e62d
md"""
### Perfilar programas

Al perfilar, se revisa el estado interno del programa en diferentes puntos. Los puntos
que aparezcan con más frecuencia son aquellos donde el programa tarda más. Esto
permite identificar cuellos de botella para optimizar. Para hacerlo basta usar el
macro `@profile`.
"""

# ╔═╡ d0e0e2d5-1c43-403e-b16b-2c6f93cdbad9
begin
	Profile.clear()
	@profile cypher_text(cesar, big_text)
	Profile.print()
end

# ╔═╡ fbea4d64-7f47-454f-9a44-98d74c222e15
md"""
## Variables globales

Acceder a variables fuera de tu alcance es muy costoso. Siempre que sea posible, hay
que mover todas las variables al alcance donde se van a usar.
"""

# ╔═╡ 5e11b816-cd4d-458d-9c67-689212fd448e
begin
	function cypher_global(letter)
		new_index = mod(cesar.pos[letter] + cesar.key - 1, length(cesar.alf)) + 1
		cesar.alf[new_index]
	end
	
	function cypher_text_global(text)
		s = ""
		for letter in text
			s = string(s, cypher_global(letter))
		end
		s
	end
end

# ╔═╡ 4a192bb9-9b0a-4981-855e-4da9bde77c61
@benchmark cypher_text_global(small_text)

# ╔═╡ f118dd0f-4bc8-4734-9de3-61353d626b2b
md"""
Podemos ver que la función que usa variables globales es en promedio tres veces más
lenta que la que recibe el cifrado como parámetro.
"""

# ╔═╡ 4819d98f-efe3-4338-9721-39fb89eca449
md"""
## Manejo de memoria

Manejar memoria de forma dinámica es muy costoso. Así que debe intentar pedir más
memoria la menor cantidad de veces posible. Hay varias técnicas para esto.
"""

# ╔═╡ 9a6340d0-2c75-4124-a871-6e107b36471d
md"""
### Alocación de memoria

Si se va a usar memoria dentro de una función, lo mejor es que esa memoria no sea
pedida dentro de la función, si no que sea un parámetro.ss
"""

# ╔═╡ c908c787-79f3-4cea-b06f-ea0c8dc20c59
md"""
### Reciclar memoria

De manera similar, si se necesita memoria, y se tiene alguna estructura que no se va a
usar más, se puede usar esa para evitar pedir más memoria.
"""

# ╔═╡ 243210f8-77b0-40a9-9dab-3d53dea74106
md"""
### Vistas

Si se necesita procesar una serie de datos, pero no es necesario modificarlos, hay una
manera de ahorrar memoria. Esto es con una vista de los datos. Esto es simplemente una
referencia de solo lectura de ellos.

Como solo es una referencia, no se necesita gran cantidad de memoria, aunque se tengan
muchos datos.
"""

# ╔═╡ 89260641-a64c-4d3d-93d4-59fc6dd7922d
md"""
### Operaciones que modifican estado

Si se tiene una operación que debe regresar una estructura similar a uno de los
parámetros pero con alguna modificación (como ordenada, con elementos nuevos o
elementos retirados) se puede modificar directamente la estrucutura original. Esto
se conoce como efectos secundarios, y es lo que hace que las funciones de Julia 
técnicamente no sean funciones en el sentido matemático, pues llamar a la misma
función varias veces con la misma estrucutra como parámetro puede dar diferentes
resuldatos.

Por lo mismo, en Julia se tiene la convención de poner un símbolo `!` al final de su
nombre, para que se tenga en cuenta de que es función no es pura.

A fin de cuentas, esto permite ahorrar memoria.
"""

# ╔═╡ 150b6e28-f317-40af-8507-75b66ecd98f4
md"""
### Alocar solo una vez

Si se requiere usar memoria dentro de una función, hay que intentar que toda esa
memoria sea pedida en un mismo punto, sin importar que se vaya a usar hasta mucho
después.
"""

# ╔═╡ d4d3f0f6-9275-4b4f-b4d5-fc6e5a4abe9e
md"""
### Estructuras estáticas
"""

# ╔═╡ 5f5cabb2-8bfc-483c-8a98-299c56750088
md"""
## Iterar
"""

# ╔═╡ 1eb2fb90-680d-4dd4-8856-99a0612fa3bf
md"""
### Broadcasting
"""

# ╔═╡ 98631876-ea33-416e-a190-0ceb8f13ea9d
md"""
### Vectorización
"""

# ╔═╡ 9a5046ee-77cf-48ff-a4c2-7e5529046e69
md"""
### Iteradores
"""

# ╔═╡ abe6052b-a18e-403e-824a-7ccdb9cfc95e
md"""
### Matrices
"""

# ╔═╡ 418da241-07f4-4043-ad90-78339b77a80a
md"""
## Sistema de tipos
"""

# ╔═╡ 8d967099-9f45-4870-a9c3-8bd53739a260
md"""
### Estabilidad de tipos
"""

# ╔═╡ abc70534-c403-435c-b3a9-cadef9e6578e
md"""
### Tipos abstractos
"""

# ╔═╡ 7ae410bb-bb9c-45f7-9b59-41527936835d
md"""
### Campos abstractos
"""

# ╔═╡ 9adf4476-4b60-4283-a697-e52e81f6a5f8
md"""
### Despache múltiple
"""

# ╔═╡ 6af25f1f-4508-4680-8397-1ddf2cbd15eb
md"""
## Algunos macros útiles
"""

# ╔═╡ c5303275-cffd-47af-9a7b-27b6d6e5b5ff
md"""
## Código intermedio
"""

# ╔═╡ 1173e0cf-d483-4cc7-bff9-7db90660413f
md"""
## Ejemplo de flujo de trabajo
"""

# ╔═╡ 0e99a6b4-54d0-445f-820e-6cbd8aea6b81
md"""
## Referencicas

* [Make Julia Code Faster](https://www.youtube.com/watch?v=S5R8zXJOsUQ)
* [Parallel Computing in Julia](https://juliaacademy.com/p/parallel-computing)
* [performance](https://docs.julialang.org/en/v1/manual/performance-tips/)
"""

# ╔═╡ Cell order:
# ╠═fef93ec6-cef7-4d2e-bf3b-2703b16b9d16
# ╠═75c847f1-e41d-4615-9b6e-dac1194785f1
# ╟─074fd91e-989c-11eb-2ee5-51c2e2315347
# ╠═0c73012e-1141-4e0f-8b88-6212b65347fb
# ╠═34ed55b5-f3cb-4207-8fc7-445e5599a88d
# ╠═d9ef4c0e-4593-4f10-aeab-8797422f9fdc
# ╠═7c49349b-6ed4-4055-8d9d-a4f1488bb46f
# ╠═0f52da77-8177-4b1b-8868-5fff10949f30
# ╠═fd838fce-ea7f-473d-8e35-ad9c3378c763
# ╠═804acee5-3755-450d-9a27-894a2451332f
# ╟─5c9352af-55ee-4031-8c80-2345698c5ee0
# ╟─0be9c339-82ba-4dda-aa87-275940e0dce5
# ╠═7cac368d-0e6d-40a2-9b26-6c8a3fcbb446
# ╠═bb69abfb-8831-43a4-ab16-dec8d8369667
# ╟─351c228e-ccc7-44e3-9007-1f09f1919daf
# ╠═c305a0b9-8ea9-4f31-ade3-dbdb39ff4686
# ╟─a2d08c5d-0572-4720-a37f-1f6ed2a5e62d
# ╠═80e0636e-8642-4ff5-ad88-04be6af2d2bf
# ╠═d0e0e2d5-1c43-403e-b16b-2c6f93cdbad9
# ╟─fbea4d64-7f47-454f-9a44-98d74c222e15
# ╠═5e11b816-cd4d-458d-9c67-689212fd448e
# ╠═4a192bb9-9b0a-4981-855e-4da9bde77c61
# ╟─f118dd0f-4bc8-4734-9de3-61353d626b2b
# ╟─4819d98f-efe3-4338-9721-39fb89eca449
# ╟─9a6340d0-2c75-4124-a871-6e107b36471d
# ╟─c908c787-79f3-4cea-b06f-ea0c8dc20c59
# ╟─243210f8-77b0-40a9-9dab-3d53dea74106
# ╟─89260641-a64c-4d3d-93d4-59fc6dd7922d
# ╟─150b6e28-f317-40af-8507-75b66ecd98f4
# ╟─d4d3f0f6-9275-4b4f-b4d5-fc6e5a4abe9e
# ╟─5f5cabb2-8bfc-483c-8a98-299c56750088
# ╟─1eb2fb90-680d-4dd4-8856-99a0612fa3bf
# ╟─98631876-ea33-416e-a190-0ceb8f13ea9d
# ╟─9a5046ee-77cf-48ff-a4c2-7e5529046e69
# ╟─abe6052b-a18e-403e-824a-7ccdb9cfc95e
# ╟─418da241-07f4-4043-ad90-78339b77a80a
# ╟─8d967099-9f45-4870-a9c3-8bd53739a260
# ╟─abc70534-c403-435c-b3a9-cadef9e6578e
# ╟─7ae410bb-bb9c-45f7-9b59-41527936835d
# ╟─9adf4476-4b60-4283-a697-e52e81f6a5f8
# ╟─6af25f1f-4508-4680-8397-1ddf2cbd15eb
# ╟─c5303275-cffd-47af-9a7b-27b6d6e5b5ff
# ╟─1173e0cf-d483-4cc7-bff9-7db90660413f
# ╟─0e99a6b4-54d0-445f-820e-6cbd8aea6b81
