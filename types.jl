### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 81c186d0-988f-11eb-386a-cfc9c0478f55
md"""
# Más cifrados

Si bien el cifrado de César es interesante, hay muchos más cifrados. Más aún, los
cifrados no están limitados a texto, si no que se pueden cifrar una gran cantidad de
datos.

Si se quisira crear un sistema que permita cifrar de manera intercambiable y flexible, tanto en el tipo de entrada como en el método de cifrado. Algo de la forma

```julia
cypher(bytes_buffer, type=:cesar, key=131)
cypher(text, type=:vigenere, key="Supercalifragilisticoespiralidoso")
```

Esto es más complicado que lo hecho anteriormente, y requiere usar todo el poder del
sistema de tipos de Julia. Así que veremos como está diseñado antes de seguir con la
implementación de los cifrados.

Hay dos grandes familiar de sistemas de tipos para lenguajes de programación
"""

# ╔═╡ 10192d0b-a614-4021-9102-e23607b3844f
md"""
* Tipos estáticos

En este mecanismo se le tiene que asignar un tipo a todos los valores desde el inicio.
Lenguajes como `C` o `Java` son de este tipo. Cada vez que se quiere utilizar un dato
se tiene que especificar su tipo. Otro lenguajes como `Haskell` o `ML` son capaces de
determinar el tipo sin necesidad de que se les indique explícitamente, pero al final
el tipo se debe saber antes de que se ejecute el código.

* Tipos dinámicos

En este caso los tipos de los datos solo se verifican en tiempo de ejecución cuando
sea estrictamente necesario. Lenguajes como `Python` o `Javascript` usan este
mecanismo.

Estrictamente hablando, Julia es un lenguaje con tipos dinámicos, pero tiene varias
caracterísiticas de lenguajes con tipos estáticos. Una caracterísitica importante es
la capacidad de anotar opcionalmente los tipos.
"""

# ╔═╡ be1cfe08-dd00-4075-a1c9-de0808d021ed
1::Int64

# ╔═╡ cf541690-5506-4b37-a95c-37ae80fe37b3
1.0::Float64

# ╔═╡ 4d2422bd-dcb6-4a54-916c-c146c0504416
"hola"::Int64

# ╔═╡ 24be241a-d249-42d1-bc48-8d8b1456f037
md"""
¡Upss! Anotamos un tipo de manera errónea.
"""

# ╔═╡ f8d39003-586b-4687-b07c-c4e239bf044f
md"""
## Tipos abstractos y concretos

El mecanismo para determinar los tipos se puede ver como una búsqueda en un árbol. En
la raíz está el tipo `Any`, los nodos internos se conocen como tipos abstractos. No se
pueden instanciar, pero sirven para clasificar y restrigir el resto de los tipos.

Se declaran de la siguiente manera.

```julia
abstract type Number end
abstract type Real <: Number end
abstract type Integer <: Real end
```

El operador `<:` indica el supertipo. Cundo no se tiene un supertipo, se asigna `Any`
como supertipo.

En las hojas están los tipos como `Int`, `Bool` o `Float64`, que representan los
posibles tipos de dato, llamados tipos concretos. Hay varios tipos de datos concretos.

Como vamos a tener varios tipo de cifrado, podría valer la pena definir un tipo
abstracto para representarlos todos.
"""

# ╔═╡ c1345ed2-2fe0-4f55-895e-af23d4158058
abstract type Cypher end

# ╔═╡ 4d53bf55-9266-4419-bea5-cd576e1c9c1a
md"""
## Tipos primitivos

Los mencionandos antes son tipos primitivos. Estos se guardan dirctamente en registros
al momento de ejecución, por lo que necesitan saber su tamaño en bits. Se declaran de
la siguiente manera

```julia
primitive type Bool <: Integer 8 end
```

Aunque probablemente todos los tipos primitivos que se quieran usar ya están declarados. 
"""

# ╔═╡ 1bb9d1e9-21fc-44e9-916c-b83925a40786
md"""
## Tipos compuestos

Sirven para juntar más tipos en una especie de registro. Son el análogo de `struct` en
`C` o una forma primitiva de clases en los lenguajes orientados a objetos.

La principal diferencia con las clases (y objetos) es que no se incluye la definición
del comportamiento en su declaración. En Julia las funciones son independientes de los
datos. Cada función puede tener diferentes muchas implementaciones, que se deciden en
tiempo de ejecución con despcho múltiple. Es uno de los principios de diseño de Julia.
"""

# ╔═╡ 34d4ff4a-ce8e-4013-be0d-e1ca06de2fbc
md"""
Aquí se podría definir todo lo que necesitaría un cifrado. En este caso podemos crear
un tipo compuesto para un cifrado de César.

En la versión anterior, además del texto, solo se necesitaba la llave en la función.
Pero esto era porque había como variables globales el alfabeto y los índices de las
letras.

Para mantener todo junto, se podría poner como atributos del tipo.
"""

# ╔═╡ 7ab12f3d-7bb1-4985-bc35-68c442b88cb3
struct CesarCypher <: Cypher
	key::Int64
	alf::Array{Char}
	pos::Dict{Char, Int64}
end

# ╔═╡ c1debed9-9754-4293-b1f1-b54ec8c8dd6d
md"""
Ahora hay que implementar el cifrado, pero usando este nuevo tipo. Como repaso la
letra ``s_{i}``, se cifra con la sguiente regla regla

``s_{i} \to s_{i+k \text{ mod } |\Sigma|}``

donde ``i`` es la posición de la letra en el alfabeto, ``k`` es la llave y ``\Sigma``
es el alfabeto.
"""

# ╔═╡ c311af2e-a36b-40db-b1cc-479ec11a9571
function cypher(cesar, letter)
	new_pos = mod(cesar.pos[letter]-1+cesar.key, length(cesar.alf))+1
	cesar.alf[new_pos]
end

# ╔═╡ 6625a474-2cf4-4537-b357-957969be0da6
c = CesarCypher(4, ['A', 'B', 'C'], Dict('A' => 1, 'B' => 2, 'C' => 3))

# ╔═╡ 0a12e55f-7212-44b7-a08d-43e80258bb40
cesar_cypher('B', c)

# ╔═╡ b4519a69-9236-4bc0-aac0-cd4df13bf578
md"""
### Constructores
"""

# ╔═╡ 15209f8b-32c9-4577-9cee-0ce09b8b942e
struct AffineCypher <: Cypher
	key::Tuple{Int64, Int64}
	alf::Array{String}
	pos::Dict{String, Int64}
end

# ╔═╡ 170d487a-eda6-40fd-8de3-19b66497b5c7
md"""
## Despacho múltiple
"""

# ╔═╡ cf599ae9-0b11-44b0-8c02-4eec5a5fa462
md"""
## Tipos paramétricos
"""

# ╔═╡ 4275ebf5-1dac-4ae4-b4a6-6b037e09892d
md"""
## Referencias
* [types](https://docs.julialang.org/en/v1/manual/types/)
* [methods](https://docs.julialang.org/en/v1/manual/methods/)
"""

# ╔═╡ Cell order:
# ╟─81c186d0-988f-11eb-386a-cfc9c0478f55
# ╟─10192d0b-a614-4021-9102-e23607b3844f
# ╠═be1cfe08-dd00-4075-a1c9-de0808d021ed
# ╠═cf541690-5506-4b37-a95c-37ae80fe37b3
# ╠═4d2422bd-dcb6-4a54-916c-c146c0504416
# ╟─24be241a-d249-42d1-bc48-8d8b1456f037
# ╟─f8d39003-586b-4687-b07c-c4e239bf044f
# ╠═c1345ed2-2fe0-4f55-895e-af23d4158058
# ╟─4d53bf55-9266-4419-bea5-cd576e1c9c1a
# ╟─1bb9d1e9-21fc-44e9-916c-b83925a40786
# ╟─34d4ff4a-ce8e-4013-be0d-e1ca06de2fbc
# ╠═7ab12f3d-7bb1-4985-bc35-68c442b88cb3
# ╟─c1debed9-9754-4293-b1f1-b54ec8c8dd6d
# ╠═c311af2e-a36b-40db-b1cc-479ec11a9571
# ╠═6625a474-2cf4-4537-b357-957969be0da6
# ╠═0a12e55f-7212-44b7-a08d-43e80258bb40
# ╟─b4519a69-9236-4bc0-aac0-cd4df13bf578
# ╠═15209f8b-32c9-4577-9cee-0ce09b8b942e
# ╟─170d487a-eda6-40fd-8de3-19b66497b5c7
# ╟─cf599ae9-0b11-44b0-8c02-4eec5a5fa462
# ╟─4275ebf5-1dac-4ae4-b4a6-6b037e09892d
