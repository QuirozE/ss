### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 4a0cfdb8-8760-11eb-0873-1bf4cbdb195d
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ 5dae1584-8760-11eb-294c-0348d7d62a13
begin
	Pkg.add("PlutoUI")
	using PlutoUI
end

# ╔═╡ 570ee268-868b-11eb-22cb-25d1a960cc58
md"""
# Un problema interesante

Un problema interesante es como comunicarse de forma segura. Una técnica para esto es la criptografía, que consiste en transforma el mensaje usando una llave, de tal forma que sea difícil obtener el mensaje original sin usar la llave. Estas transformaciones se llaman cifrados.
"""

# ╔═╡ 93c9a7e2-8762-11eb-1301-29befdbc7615
md"""
## Cifrado de César

Uno de los cifrados más antiguos es el cifrado de César. Este fue usado por el emperador romano Julio César durante la conquista de Galia (lo que hoy es Francia).

Este consiste en transforma el alfabeto con un desplazamiento.

Por ejemplo, si se quisiera desplazar el alfabeto por tres letras se tendría el siguiente cifrado

| Original | A | B | C | D | E |$\dots$ | V | W | X | Y | Z |
|----------|---|---|---|---|---|--------|---|---|---|---|---|
|Cifrado   | D | E | F | G | H |$\dots$ | Y | Z | A | B | C |

En este cifrado, la llave es el número de deplazamiento. En este caso `key = 3`.

Si se conoce el mensaje cifrado, basta con recorrer el alfabeto en dirección opuesta. Es un cifrado bastante sencillo.

Pero antes de ver los detalles de implementación, hay que introducir las cosas más básicas de Julia.

"""

# ╔═╡ 08dc8a82-8776-11eb-3f1a-2b28aba9fd95
md"""
## Números

Pueden ser enteros o decimales. Hay muchos más tipos, pero para el cifrado basta con estos.
"""

# ╔═╡ 45cb4852-8776-11eb-2291-c546e2fe7462
879

# ╔═╡ 492a3fbc-8776-11eb-3404-eb9c9b4343f1
9898.987

# ╔═╡ ebc8d1a2-8776-11eb-3c12-b1946911f134
md"""
### Operaciones básica

Ahora que ya están los valores del texto a cifrar y la llave, hay que ver que tipo de operaciones se pueden hacer.

Para números, están las operaciones básicas

| Operación | Código |
|:---------:|:------:|
|$x+y$      |`x+y`   |
|$x-y$      |`x+y`   |
|$x\times y$|`x*y`   |
|$x/y$      |`x/y`   |
|$x$ mod $y$|`x%y`   |
|$x^y$      |`x^y`   |

Hay muchos más, pero se pueden aprender cuando se necesiten.

"""

# ╔═╡ 9761bfe8-8776-11eb-07ea-3f1569e3ad94
md"""
## Cadenas

También vamos a necesitar cadenas de texto. Estas se delimitan con "comillas dobles" y pueden contener cualquier caracter (excepto comillas dobles). Alternativamente se pueden usar \"""comillas triples""\" para usar comillas dentro del texto.
"""

# ╔═╡ c3a4b4f2-8776-11eb-3e07-410b48bb051b
"hola"

# ╔═╡ 2296beea-8777-11eb-1482-938fe32607cd
"""la llave podría ser "b" ¿no?"""

# ╔═╡ 87c407f2-8777-11eb-2355-edd7cf9f535e
md"""
### Operaciones básicas
"""

# ╔═╡ 102190c4-8778-11eb-08a1-d3ba78f37a46
md"""
#### Concatenación

La primera operación sería la concatenación. Hay varias maneras de hacer esto.

Primero con el operador `*`.
"""

# ╔═╡ b0cb8cd8-8777-11eb-0791-bb195b4e4cd4
"Hola " * "mundo"

# ╔═╡ bb0151ba-8777-11eb-0028-8f9974bafaba
md"""
O con la función `string`. La diferencia es que los parámetros de esta función no necesariamente son cadenas.
"""

# ╔═╡ f39f43ec-8777-11eb-23b0-db1d2d53b579
string("La llave es ", 3)

# ╔═╡ fe25deb6-8777-11eb-395c-e7b6a72e5b74
md"""
#### Interpolación
Otra operación sobre cadenas (muy relacionada a la concatenación) es la interpolación. Esto es insertar una expresión que se pueda evaluar a un valor dentro de una cadena. Para indicar que partes se va a insertar, se debe poner antes un `$`.
"""

# ╔═╡ efdac270-876f-11eb-0a14-c75fd8e64f57
md"""
## Variables

Es necesario guardar los datos intermedios. Para esto se usan variables. En Julia, las variables pueden ser casi lo que sea excepto palabras reservadas. A grandes rasgos, deben iniciar con una letra Unicode, y pueden estar seguido de casi cualquier cosas. Más detalles en la [documentación oficial](https://docs.julialang.org/en/v1/manual/variables/).

Se declaran con la siguiente sintaxis.

```julia
var = val
```
"""

# ╔═╡ 4eb98a80-8778-11eb-25d6-8d3c5f6d14fb
"2 elevado a la 6 potencia es $(2^6)"

# ╔═╡ 4be651f0-8771-11eb-2a26-3b461e6f5095
mensaje_secreto = "Nadie se puede enterar"

# ╔═╡ 52a9b1f8-8771-11eb-0444-93d3a2217e19
llave = 3

# ╔═╡ b8c0dafe-8779-11eb-0704-abb693c60b0e
md"""
Ya tenemos la llave y el mensaje a cifrar. Hay que crear el mensaje cifrado desplazando todas las letras del mensaje original. Pero, ¿cómo sabemos siquiera el orden sobre el cuál hay que desplazar?

Para esto se puede introducir un nuevo tipo de valor
"""

# ╔═╡ 115b5a86-877a-11eb-0bd8-235fff98e380
md"""
## Arreglos

Estos son un tipo de estrucutras de datos que permiten guardar valores en una secuencia. Se defininen poniendo los elementos entre corchetes
"""

# ╔═╡ 6c1feb1c-877a-11eb-0413-a75610ec89be
arreglo = [1, "hola", 9.0]

# ╔═╡ 893d8844-877a-11eb-3765-6326a08daf2c
md"""
### Operaciones básicas

Para acceder a los elementos del arreglo, se usa el índice del elementos deseado entre corchetes
"""

# ╔═╡ ac232512-877a-11eb-3745-9713e5b283db
arreglo[2]

# ╔═╡ 9523a140-877d-11eb-1763-a78a40f69274
md"""
Para cambiar un elemento se accede a el, y se asigna como si fuera una variable
"""

# ╔═╡ ad81c79e-877d-11eb-3583-af4e32b59111
arreglo[2] = "¿cómo estás?"

# ╔═╡ b9d93964-877d-11eb-0d1c-ab5736be9b72
arreglo

# ╔═╡ f6e1759a-877c-11eb-06d9-2543b9334985
md"""
Para agregar un elemento se usa la función `push!`. Cuidado, porque esta función modifica la estructura, así que llamarla varias veces con los mismos parámetros no tendrá el mismo resultado.
"""

# ╔═╡ 726017b0-877d-11eb-345b-e3d86663f976
push!(arreglo, 7)

# ╔═╡ 3bb647e2-877e-11eb-291f-6fa3e342a75d
md"""
Y para eliminar el último elemento agregado, se puede usar `pop!`
"""

# ╔═╡ 4ad56d8e-877e-11eb-181c-ab5134619b01
pop!(arreglo)

# ╔═╡ 58936642-877e-11eb-1ad1-b7c15a8c5eb8
arreglo

# ╔═╡ b4ec5e66-877a-11eb-2b2e-7174ed5161ef
md"""
Con esto, se puede definir el orden del alfabeto
"""

# ╔═╡ c350ff84-877a-11eb-0473-59b88beda3a3
alfabeto = [
	"A", "B", "C", "D", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Ñ", "O", "P", 
	"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# ╔═╡ 5378cd9e-877b-11eb-03ab-670b9ae5a144
md"""
Ahora, para obtener la letra desplazada, basta con encontrar su índice en el alfabeto, sumarle 3 y tomar el valor del arreglo en esa posición.

Pero ¿cómo se obtiene el índice de una letra? Se necesita algo similar al opuesto de un arreglo. Algo que te pueda obtener los índices usando los valores.
"""

# ╔═╡ 2661e210-877f-11eb-1608-030d3b914161
md"""
## Diccionarios

Son estructuras similares a los arreglos, pero en lugar de índices se puede acceder a los valores usando (casi) cualquier cosa. Como muchas cosas no están ordenadas, entonces los diccionarios no definien un orden.

Se declaran con el constructor `Dict`, junto con las parejas de llave valor `key => val`.
"""

# ╔═╡ 8b09f194-877f-11eb-00e6-85a943d5a4da
dict = Dict("a" => 14, 5 => "abalone")

# ╔═╡ 9f24151a-877f-11eb-0a53-9f40a74c94ee
md"""
Se acceden y modifican elementos con la misma sintaxis que los arreglos.

Cabe notar que como no tienen orden, agregar y quitar elementos es un poco diferentes. Para agregar un nuevo elemento basta con asignarlo.
"""

# ╔═╡ c2c41b78-877f-11eb-3868-3fb028f3fed3
dict["hola"] = "mundo"

# ╔═╡ c8989ab0-877f-11eb-3399-61eb6c07f381
dict

# ╔═╡ deffc486-877f-11eb-30ec-af84859d9167
md"""
Para quitarlos está la función `delete!`.
"""

# ╔═╡ eb50652e-877f-11eb-0429-09e00c13b457
delete!(dict, "hola")

# ╔═╡ 059a3554-8780-11eb-333e-a17420760cf6
md"""
Entonces se puede definir un diccionario para obtener el índice de las letras.
"""

# ╔═╡ 1ccd0616-8780-11eb-092e-ddeadb003dec
indices_alfabeto = Dict(
	"A" => 1, "B" => 2, "C" => 3, "D" => 4, "E" => 5, "F" => 6, "G" => 7, "H" => 8,
	"I" => 9, "J" => 10, "K" => 11, "L" => 12, "M" => 13, "N" => 14, "Ñ" => 15,
	"O" => 16, "P" => 17, "Q" => 18, "R" => 19, "S" => 20, "T" => 21, "U" => 22,
	"V" => 23, "W" => 24, "X" => 25, "Y" => 26, "Z" => 27
)

# ╔═╡ 1a003e22-8781-11eb-10ec-a7987098909b
md"""
Ahora ya se podría obtener el cifrad de alguna letra
"""

# ╔═╡ 25dafc1c-8781-11eb-3554-955333b6ab9c
A_crifrada = alfabeto[indices_alfabeto["A"]+llave]

# ╔═╡ Cell order:
# ╠═4a0cfdb8-8760-11eb-0873-1bf4cbdb195d
# ╠═5dae1584-8760-11eb-294c-0348d7d62a13
# ╟─570ee268-868b-11eb-22cb-25d1a960cc58
# ╟─93c9a7e2-8762-11eb-1301-29befdbc7615
# ╟─08dc8a82-8776-11eb-3f1a-2b28aba9fd95
# ╠═45cb4852-8776-11eb-2291-c546e2fe7462
# ╠═492a3fbc-8776-11eb-3404-eb9c9b4343f1
# ╟─ebc8d1a2-8776-11eb-3c12-b1946911f134
# ╟─9761bfe8-8776-11eb-07ea-3f1569e3ad94
# ╠═c3a4b4f2-8776-11eb-3e07-410b48bb051b
# ╠═2296beea-8777-11eb-1482-938fe32607cd
# ╟─87c407f2-8777-11eb-2355-edd7cf9f535e
# ╟─102190c4-8778-11eb-08a1-d3ba78f37a46
# ╠═b0cb8cd8-8777-11eb-0791-bb195b4e4cd4
# ╟─bb0151ba-8777-11eb-0028-8f9974bafaba
# ╠═f39f43ec-8777-11eb-23b0-db1d2d53b579
# ╟─fe25deb6-8777-11eb-395c-e7b6a72e5b74
# ╟─efdac270-876f-11eb-0a14-c75fd8e64f57
# ╠═4eb98a80-8778-11eb-25d6-8d3c5f6d14fb
# ╠═4be651f0-8771-11eb-2a26-3b461e6f5095
# ╠═52a9b1f8-8771-11eb-0444-93d3a2217e19
# ╟─b8c0dafe-8779-11eb-0704-abb693c60b0e
# ╟─115b5a86-877a-11eb-0bd8-235fff98e380
# ╠═6c1feb1c-877a-11eb-0413-a75610ec89be
# ╟─893d8844-877a-11eb-3765-6326a08daf2c
# ╠═ac232512-877a-11eb-3745-9713e5b283db
# ╟─9523a140-877d-11eb-1763-a78a40f69274
# ╠═ad81c79e-877d-11eb-3583-af4e32b59111
# ╠═b9d93964-877d-11eb-0d1c-ab5736be9b72
# ╟─f6e1759a-877c-11eb-06d9-2543b9334985
# ╠═726017b0-877d-11eb-345b-e3d86663f976
# ╟─3bb647e2-877e-11eb-291f-6fa3e342a75d
# ╠═4ad56d8e-877e-11eb-181c-ab5134619b01
# ╠═58936642-877e-11eb-1ad1-b7c15a8c5eb8
# ╟─b4ec5e66-877a-11eb-2b2e-7174ed5161ef
# ╠═c350ff84-877a-11eb-0473-59b88beda3a3
# ╟─5378cd9e-877b-11eb-03ab-670b9ae5a144
# ╟─2661e210-877f-11eb-1608-030d3b914161
# ╠═8b09f194-877f-11eb-00e6-85a943d5a4da
# ╟─9f24151a-877f-11eb-0a53-9f40a74c94ee
# ╠═c2c41b78-877f-11eb-3868-3fb028f3fed3
# ╠═c8989ab0-877f-11eb-3399-61eb6c07f381
# ╟─deffc486-877f-11eb-30ec-af84859d9167
# ╠═eb50652e-877f-11eb-0429-09e00c13b457
# ╟─059a3554-8780-11eb-333e-a17420760cf6
# ╠═1ccd0616-8780-11eb-092e-ddeadb003dec
# ╟─1a003e22-8781-11eb-10ec-a7987098909b
# ╠═25dafc1c-8781-11eb-3554-955333b6ab9c
