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

Si se quisira crear un sistema que permita cifrar de manera intercambiable y flexible,
tanto en el tipo de entrada como en el método de cifrado. Algo de la forma

```julia
cypher(bytes_buffer, type=:cesar, key=131)
cypher(text, type=:affine, key=(1, 2))
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
del procesador al momento de ejecución, por lo que necesitan saber su tamaño en bits.
Se declaran de la siguiente manera

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

```julia
struct CesarCypher <: Cypher
	key::Int64
	alf::Array{Char}
	pos::Dict{Char, Int64}
end
```
"""

# ╔═╡ 588798f9-4e75-4f0f-bbf2-39641f6de0bd
md"""
Ahora que se tiene definido el tipo compuesto, ¿cómo se crea?
"""

# ╔═╡ b2a19d5f-3279-4d7f-87cd-f91c67cc4c1c
md"""
### Constructores
"""

# ╔═╡ 9fc0a41b-9ed6-4e6a-8d42-cb75930da59d
md"""
Cada tipo compuesto implícitamente define una función del mismo nombre del tipo,
donde los parámetros son los atributos del tipo y donde el valor de retorno es uno 
nuevo valor del tipo compuesto. Esta función se llama constructor.

Este constructor se usaría de la siguiente manera

```julia
cypher = CesarCypher(0, [], Dict())
```
"""

# ╔═╡ de63bf2e-a5dd-47bf-8c84-76e63b92f49f
md"""
Esto puede ser suficiente en muchos casos, pero se puede notar que las posiciones 
se pueden generar usando el alfabeto, sin necesidad de pasarlo explícitamente. Sin
embargo, el constructor por omisión debe recibir todos los atributos.

Pero al ser solo una función, se puede crear otra función constructor que haga lo que
queremos. Estas funciones extra se llaman constructores externos. Deben tener el mismo
nombre que el tipo compuesto, y regresar una instancia del tipo compuesto que deben
generar usando otros constructores (como el constuctor por omisión).

Por ejemplo, para solo recibir la llave y el alfabeto y generar las posiciones
automáticamente se puede usar el siguiente construtor

```julia
function CesarCypher(key, alf)
	pos = Dict()
	for (i, letter) in enumerate(alf)
		por[letter] = i
	end
	CesarCypher(ket, alf, pos)
end
```

Por lo que la definición final del tipo sería la siguiente
"""

# ╔═╡ c30f6738-57e4-4918-a64f-f0b38dc1b16d
begin
	struct CesarCypher <: Cypher
		key::Int64
		alf::Array{Char}
		pos::Dict{Char, Int64}
	end
	
	function CesarCypher(key, alf)
		pos = Dict()
		for (i, letter) in enumerate(alf)
			por[letter] = i
		end
		CesarCypher(ket, alf, pos)
	end
end

# ╔═╡ 87a6b719-e24e-4fec-9265-3757ee960dc8
md"""
¿Esto es suficiente para inicializar los valores de los tipos compuestos? Hay algunos
casos más. Para esto, vamos a ver otro tipo de cifrado un poco más complicado: los
cifrados afines. Los detalles del proceso de cifrado se verán más adelante, pero su
llave consta de dos primos relativos. El tipo compuesto para este cifrado podría ser
el siguiente

```julia
struct AffineCypher <: Cypher
	key::Tuple{Int64, Int64}
	alf::Array{Char}
	pos::Dict{Char, Int64}
end
```
"""

# ╔═╡ 9da18c8c-fd0b-4309-b366-c3f537742560
md"""
¿Sería posible verificar esto al crear el tipo? Claramente no es posible usando
constructores externos, pues estos simplemente añaden otro tipo de constructor, no
pueden modificar los que ya existen.

Para esto se tienen que usar constructores internos. Estos remplazan al constructor
por omisión. Deben ser declarados dentro del cuerpo del `struct` al que pertenencen.

En estos, se crea una nueva instancia del tipo usando la función `new` que no 
necesariamente inicializa los atributos. Esto resulta útil para tipos recursivos.
"""

# ╔═╡ 88c36bda-e2f5-48a1-a39b-cbac20fd9608
md"""
Un constructor interno que implemente la restricción de los cifrados afines podría ser
el siguiente

```julia
function AffineCypher(key, alf, pod)
	if gcd(key[1], key[2]) != 1
		error("Non-reversible affine transformation")
	end
	new(key, alf, pos)
end
```

Por lo que la declaración completa del tipo sería
"""

# ╔═╡ 4b0894c4-fad8-4d56-b8d2-2ba4db516c29
begin
	struct AffineCypher <: Cypher
		key::Tuple{Int64, Int64}
		alf::Array{Char}
		pos::Dict{Char, Int64}

		function AffineCypher(key, alf, pod)
			if gcd(key[1], key[2]) != 1
				error("Non-reversible affine transformation")
			end
			new(key, alf, pos)
		end
	end
	
	function AffineCypher(key, alf)
		pos = Dict()
		for (i, letter) in enumerate(alf)
			por[letter] = i
		end
		CesarCypher(ket, alf, pos)
	end
end

# ╔═╡ 170d487a-eda6-40fd-8de3-19b66497b5c7
md"""
## Despacho múltiple
"""

# ╔═╡ 2c2d5df6-891d-4487-b37f-9ee31442dff0
md"""
Ahora hay que implementar el cifrado, pero usando este nuevo tipo. Como repaso la
letra ``s_{i}``, se cifra con la siguiente regla regla

``s_{i} \to s_{i+k \text{ mod } |\Sigma|}``

donde ``i`` es la posición de la letra en el alfabeto, ``k`` es la llave y ``\Sigma``
es el alfabeto.
"""

# ╔═╡ 089ec0de-3af3-463e-83c8-b9b140df0859
function cypher(cesar, letter)
	new_pos = mod(cesar.pos[letter]-1+cesar.key, length(cesar.alf))+1
	cesar.alf[new_pos]
end

# ╔═╡ 99619da2-9bfb-44cb-8512-a3deeabb5962
md"""
Ahora hay que agregar el cifrado afín. En este, determinar la letra a la que se cifra
no solo es un desplazamiento, pero una transformación afín. En ``\mathbb{Z}``, estas
transformaciones son de la forma

``
f(x) = xk_{1} + k_{2}
``

Así que la regla para este cifrado sería

``
s_{i} \to s_{ik_{1}+k_{2} \text{ mod } |\Sigma|}
``

Hay que notar que no todas las transformaciones afines son reversibles, por lo que no
todas las llaves ``k_{1}, k_{2}`` representan cifrados válidos. Como se dijo
anteriormente, la tranformación es reversible si las dos partes de la llave son primos
relativos.
"""

# ╔═╡ cf599ae9-0b11-44b0-8c02-4eec5a5fa462
md"""
## Tipos paramétricos
"""

# ╔═╡ 4275ebf5-1dac-4ae4-b4a6-6b037e09892d
md"""
## Referencias
* [types](https://docs.julialang.org/en/v1/manual/types/)
* [constructors](https://docs.julialang.org/en/v1/manual/constructors/)
* [methods](https://docs.julialang.org/en/v1/manual/methods/)
* [affine](https://brilliant.org/wiki/affine-transformations/)
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
# ╟─588798f9-4e75-4f0f-bbf2-39641f6de0bd
# ╟─b2a19d5f-3279-4d7f-87cd-f91c67cc4c1c
# ╟─9fc0a41b-9ed6-4e6a-8d42-cb75930da59d
# ╟─de63bf2e-a5dd-47bf-8c84-76e63b92f49f
# ╠═c30f6738-57e4-4918-a64f-f0b38dc1b16d
# ╟─87a6b719-e24e-4fec-9265-3757ee960dc8
# ╟─9da18c8c-fd0b-4309-b366-c3f537742560
# ╟─88c36bda-e2f5-48a1-a39b-cbac20fd9608
# ╠═4b0894c4-fad8-4d56-b8d2-2ba4db516c29
# ╟─170d487a-eda6-40fd-8de3-19b66497b5c7
# ╟─2c2d5df6-891d-4487-b37f-9ee31442dff0
# ╠═089ec0de-3af3-463e-83c8-b9b140df0859
# ╠═99619da2-9bfb-44cb-8512-a3deeabb5962
# ╟─cf599ae9-0b11-44b0-8c02-4eec5a5fa462
# ╠═4275ebf5-1dac-4ae4-b4a6-6b037e09892d
