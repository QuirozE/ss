### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 570ee268-868b-11eb-22cb-25d1a960cc58
md"""
# Cifrado de César

Un problema interesante es como comunicarse de forma segura. Una técnica para esto es
la criptografía, que consiste en transforma el mensaje usando una llave, de tal forma
que sea difícil obtener el mensaje original sin usar la llave. Estas transformaciones
se llaman cifrados.
"""

# ╔═╡ 93c9a7e2-8762-11eb-1301-29befdbc7615
md"""

Uno de los cifrados más antiguos es el cifrado de César. Este fue usado por el 
emperador romano Julio César durante la conquista de Galia (lo que hoy es Francia).

Este consiste en transforma el alfabeto con un desplazamiento.

Por ejemplo, si se quisiera desplazar el alfabeto por tres letras se tendría el 
siguiente cifrado

| Original | A | B | C | D | E |$\dots$ | V | W | X | Y | Z |
|----------|---|---|---|---|---|--------|---|---|---|---|---|
|Cifrado   | D | E | F | G | H |$\dots$ | Y | Z | A | B | C |

En este cifrado, la llave es el número de deplazamiento. En este caso `key = 3`.

Si se conoce el mensaje cifrado, basta con recorrer el alfabeto en dirección opuesta. 
Es un cifrado bastante sencillo.

Pero antes de ver los detalles de implementación, hay que introducir las cosas más 
básicas de Julia.

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
|$x^y$      |`x^y`   |

Hay muchos más, pero se pueden aprender cuando se necesiten.

"""

# ╔═╡ 9761bfe8-8776-11eb-07ea-3f1569e3ad94
md"""
## Cadenas

También vamos a necesitar cadenas de texto. Las cadenas son secuencias de caracteres. Los characteres cualquier símbolo Unicode.

Lo caracteres se denota con comillas simples
"""

# ╔═╡ d8e553d5-08c6-4cc1-90c1-a1b28b270752
'a'

# ╔═╡ 3e63a610-929e-4915-833d-c35318d35e5a
md"""
Las cadenas se delimitan con "comillas dobles" y pueden contener cualquier caracter
(excepto comillas dobles). Alternativamente se pueden usar \"""comillas triples""\" 
para usar  comillas dentro del texto.
"""

# ╔═╡ c3a4b4f2-8776-11eb-3e07-410b48bb051b
"hola"

# ╔═╡ 2296beea-8777-11eb-1482-938fe32607cd
"""la llave podría ser "b" ¿no?"""

# ╔═╡ 87c407f2-8777-11eb-2355-edd7cf9f535e
md"""
### Operaciones básicas
"""

# ╔═╡ 352bd483-0561-4686-8979-e90b0c418a3f
md"""
### Acceso a elementos

Se puede acceder a los elementos de una cadena usando corchetes. Como se mencionó
anteriormente, los elemento de una cadena son un caracter Unicode.
"""

# ╔═╡ d84f9a56-9e86-4784-9b39-8246ad099973
"hola"[1]

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
Otra operación sobre cadenas (muy relacionada a la concatenación) es la interpolación. 
Esto es insertar una expresión que se pueda evaluar a un valor dentro de una cadena. 
Para indicar que partes se va a insertar, se debe poner antes un `$`.
"""

# ╔═╡ 14200a8c-33da-43bc-840b-c9ba38cf3848
"2 elevado a la 6 potencia es $(2^6)"

# ╔═╡ efdac270-876f-11eb-0a14-c75fd8e64f57
md"""
## Variables

Es necesario guardar los datos intermedios. Para esto se usan variables. En Julia, las 
variables pueden ser casi lo que sea excepto palabras reservadas. A grandes rasgos, 
deben iniciar con una letra Unicode, y pueden estar seguido de casi cualquier cosas. 
Más detalles en la [documentación oficial]
(https://docs.julialang.org/en/v1/manual/variables/).

Se declaran con la siguiente sintaxis.

```julia
var = val
```
"""

# ╔═╡ 4be651f0-8771-11eb-2a26-3b461e6f5095
mensaje_secreto = "Nadie se puede enterar"

# ╔═╡ 52a9b1f8-8771-11eb-0444-93d3a2217e19
llave = 3

# ╔═╡ b8c0dafe-8779-11eb-0704-abb693c60b0e
md"""
Ya tenemos la llave y el mensaje a cifrar. Hay que crear el mensaje cifrado 
desplazando todas las letras del mensaje original. Pero, ¿cómo sabemos siquiera el 
orden sobre el cuál hay que desplazar?

Para esto se puede introducir un nuevo tipo de valor
"""

# ╔═╡ 115b5a86-877a-11eb-0bd8-235fff98e380
md"""
## Arreglos

Estos son un tipo de estrucutras de datos que permiten guardar valores en una 
secuencia. Se defininen poniendo los elementos entre corchetes
"""

# ╔═╡ 6c1feb1c-877a-11eb-0413-a75610ec89be
arreglo = [1, "hola", 9.0]

# ╔═╡ 893d8844-877a-11eb-3765-6326a08daf2c
md"""
### Operaciones básicas

Para acceder a los elementos del arreglo, se usa el índice del elementos deseado entre 
corchetes
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
Para agregar un elemento se usa la función `push!`. Cuidado, porque esta función 
modifica la estructura, así que llamarla varias veces con los mismos parámetros no 
tendrá el mismo resultado.
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
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 
	'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# ╔═╡ 5378cd9e-877b-11eb-03ab-670b9ae5a144
md"""
Ahora, para obtener la letra desplazada, basta con encontrar su índice en el alfabeto, 
sumarle 3 y tomar el valor del arreglo en esa posición.

Pero ¿cómo se obtiene el índice de una letra? Se necesita algo similar al opuesto de 
un arreglo. Algo que te pueda obtener los índices usando los valores.
"""

# ╔═╡ 2661e210-877f-11eb-1608-030d3b914161
md"""
## Diccionarios

Son estructuras similares a los arreglos, pero en lugar de índices se puede acceder a 
los valores usando (casi) cualquier cosa. Como muchas cosas no están ordenadas, 
entonces los diccionarios no definien un orden.

Se declaran con el constructor `Dict`, junto con las parejas de llave valor `key => 
val`.
"""

# ╔═╡ 8b09f194-877f-11eb-00e6-85a943d5a4da
dict = Dict("a" => 14, 5 => "abalone")

# ╔═╡ 9f24151a-877f-11eb-0a53-9f40a74c94ee
md"""
Se acceden y modifican elementos con la misma sintaxis que los arreglos.

Cabe notar que como no tienen orden, agregar y quitar elementos es un poco diferentes. 
Para agregar un nuevo elemento basta con asignarlo.
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
	'A' => 1, 'B' => 2, 'C' => 3, 'D' => 4, 'E' => 5, 'F' => 6, 'G' => 7, 'H' => 8,
	'I' => 9, 'J' => 10, 'K' => 11, 'L' => 12, 'M' => 13, 'N' => 14, 'Ñ' => 15,
	'O' => 16, 'P' => 17, 'Q' => 18, 'R' => 19, 'S' => 20, 'T' => 21, 'U' => 22,
	'V' => 23, 'W' => 24, 'X' => 25, 'Y' => 26, 'Z' => 27
)

# ╔═╡ 1a003e22-8781-11eb-10ec-a7987098909b
md"""
Ahora ya se podría obtener el cifrad de alguna letra
"""

# ╔═╡ 25dafc1c-8781-11eb-3554-955333b6ab9c
A_crifrada = alfabeto[indices_alfabeto['A']+llave]

# ╔═╡ 56fbf961-8e34-4838-8fbc-8dcf92251faf
md"""
Ahora, tener que escribir `alfabeto[indices_alfabeto[letra]+llave]` cada vez que se 
quiera cifrar una letra sería tedioso. Para esto se puede definir una función que lo 
haga.
"""

# ╔═╡ 2ab0e75b-b07b-48d6-8dc4-413ab838af82
md"""
## Funciones

Todas las acciones que manipulan datos vistas hasta ahora son funciones de Julia. 
Estas permiten encapsular cierto comportamiento para no repetir código. Son
similares a las funciones matemáticas. Por ejemplo, para ``f(x) \to x^{2}+2``, la
sintaxis para declararla sería
"""

# ╔═╡ 8be85ec1-af78-4f88-870e-2d3950959265
function f1(x)
	x^2+2
end

# ╔═╡ bae9fafe-2187-46c6-89f2-332c6f4eb689
md"""
El valor en la última línea es el valor que regresa la función. Para funciones
sencillas (de solo una línea) existe una notación más compacta similar a la de las 
funciones matemáticas
"""

# ╔═╡ 90e27d28-bb92-4318-af52-d95d79cd8c1f
f2(x) = x^2+2

# ╔═╡ cc6cf9c2-87c1-40d6-8f70-920e143eeeb4
md"""
Como las funciones matemáticas, pueden tomar valores de un conjunto (tipo) arbitrario
(posiblemente varios valores en forma de tupla) y regresar valores de cualquier
conjunto (tipo), incluído un tipo `Nothing`, que posee un solo valor `nothing` que
representa la ausencia de valor a regresar.

La gran diferencia con las funciones matemáticas es que las funciones en Julia pueden
modificar el estado del programa, por lo que llamaras varias veces con los mismos
argumentos puede dar resultados diferentes.
"""

# ╔═╡ e7ef2738-0d04-4726-8750-6d3af60e1c60
md"""
Así, se puede esribir el cifrado anterior como una función que reciba la letra a 
cifrar junto con la llave y devuelva la letra cifrada.
"""

# ╔═╡ 0a794f1e-bae0-48bc-8094-c5700c029c19
cesar_cypher_v0(letra, llave) = alfabeto[indices_alfabeto[letra]+llave]

# ╔═╡ 73841088-e2f4-4357-8eb4-e479a3da045b
cesar_cypher_v0('A', 1)

# ╔═╡ 3e0501bf-45f9-4b04-a30c-ab844e40b24a
md"""
Ahora probemos con más valores
"""

# ╔═╡ 2f56e4f3-f648-4742-aa97-df8dedec1650
cesar_cypher_v0('Z', 1)

# ╔═╡ 387b9b26-6e63-464b-865b-a8ba23c2e485
md"""
¡Upps! La función no toma en cuenta que el desplazamiento se cicla. ¿Cómo se arregla
esto?
"""

# ╔═╡ 42be3307-ebe3-4787-83c0-b12f30f70109
md"""
## Ejecucioń condicional

Hay que decidir cuando se cicla un desplazamiento y tomar acciones diferentes si es el
caso. Como las acciones pueden cambiar dependiendo de los datos, se modifica el flujo
del programa.

Para esto se puede usar ejecuciones condicionales. Tienes la sintaxis de
"""

# ╔═╡ ac03c2c4-b6f5-4d44-a424-72b12a9a05a0
if true
	5
else
	4
end

# ╔═╡ 126d3903-6fad-4419-8d13-e0f2ebb98b91
md"""
Y se pueden encadenar varias condiciones de forma similar
"""

# ╔═╡ b483cfc0-e4db-4750-8999-f9d8e02ee797
if 1 < 0
	"hola"
elseif 9 > 10
	"como"
elseif 8 < 9
	"estás"
else
	"?"
end

# ╔═╡ e25f1b55-cc54-4f53-8137-d4506a34a61c
md"""
### Expresiones booleanas

Las condiciones deben ser expresiones que se puedan evaluar a `True` o `False`.
Para números, existen los operadores de comparación

| Operación   | Código           |
|:-----------:|:----------------:|
|``x<y``      |`x<y`             |
|``x\le y``   |`x≤y` o `x <= y`  |
|``x>y``      |`x>y`             |
|``x\ge y``   |`x≥y` o `x >= y`  |

Y en general para cualquier valor, están los operadores de igualdad

| Operación   | Código         |
|:-----------:|:--------------:|
|``x=y``      |`x==y`          |
|``x\ne y``   |`x!=y` o `x≠y`  |

Además, si ya se tiene un valor booleano, existen operadores para manipularlos.

| Operación    | Código         |
|:------------:|:--------------:|
|``p\lor q``   |`p\|\|q`        |
|``p \land q`` |`p&&q` o `x≠y`  |
|``\lnot p``   |`!p` o `x≠y`    |

Más en general, cualquier función que regrese un valor booleano se puede usar como
condición.

"""


# ╔═╡ 35a9372b-2a1c-44c3-b69a-f5ba60d3075d
md"""
Entonces, la función de cifrado se puede reescribir usando estos. Como ya no será de
una línea, sería más cómodo reescribirla con la otra notación
"""

# ╔═╡ e854477e-61a9-4d2d-8252-9634e28beaa2
function cesar_cypher_v1(letra, llave)
	alf_l = length(alfabeto)
	nuevo_indice = indices_alfabeto[letra] + llave
	
	if nuevo_indice > alf_l
		nuevo_indice -= alf_l
	elseif nuevo_indice < 1
		nuevo_indice += alf_l
	end
	
	alfabeto[nuevo_indice]
end

# ╔═╡ 3b21bc0f-e7f0-423f-8b80-92f48b4d35da
cesar_cypher_v1('Z', 1)

# ╔═╡ 2ec31d4c-f08e-4f5a-9f0a-d530350c8623
md"""
Este problema del desplazamientom fue más una escusa para introducir expresiones
condicionales. El mismo efecto se puede obtener de manera más sencilla y eficiente
usando aritmética modular.
"""

# ╔═╡ a0c45f7d-00ef-45ef-ae2e-5527048985a2
function cesar_cypher_v2(letra, llave)
	alf_l = length(alfabeto)
	nuevo_indice = mod(indices_alfabeto[letra] + llave - 1, alf_l) + 1
	alfabeto[nuevo_indice]
end

# ╔═╡ 8e4cf0f3-84a7-4622-b2e4-064d6a0ac09b
cesar_cypher_v2('Z', 1)

# ╔═╡ 67fdce81-e5ab-412f-8ac9-99f6f16abae7
md"""
Todo funciona a la perfección
"""

# ╔═╡ da0124e3-8504-4f0c-a0bd-2d35f07fa34e
md"""
## Ciclos

Ahora, ya se puede cifrar una letra. ¿Cómo se cifraría un texto, i.e. muchas letras?
En Julia hay varios mecanismos para repetir operaciones sobre muchos valores. Los más
sencillos son los ciclos.

El ciclo `while` simplemente repite una acción mientras la condición sea verdadera.
"""

# ╔═╡ b561a89e-72fe-4e06-88eb-dd4fdb7ea7d9
begin
	l = []
	while length(l) < 10
		push!(l, 10)
	end
	l
end

# ╔═╡ daef06cb-b9d8-4b27-8f59-82745fd9ecb9
md"""
El ciclo `for` permite trabajar con los elementos de una colección. La colección tiene
que ser iterable.
"""

# ╔═╡ e554f82c-8127-4c20-a6f0-458f24007532

begin
	sum_prevs = [0]
	for i in 2:10
		push!(sum_prevs, sum_prevs[i-1]+i)
	end
	sum_prevs
end
	

# ╔═╡ f61e722d-c34f-4ba4-98f4-d962792323ea
md"""
Como algo es iterable es un tema un poco más avanzado. En el caso anterior, `2:10`
es un rango, y al iterarlo se obtienen los números en ese rango. Como otros ejemplos,
los arreglos pueden iterar sobr sus elementos y las cadenas sobre sus caracteres.
"""

# ╔═╡ f5153530-8d12-453a-86c8-f9a4e7803a7e
md"""
Así que se podría escribir una función para cifrar un texto usando un ciclo `for`
para iterar sobre el texto. Se utiliza la función `string` para contactenar porque
`cesar_cypher` devuelve caracteres, no cadenas.
"""

# ╔═╡ 0a379cce-43e3-4cfa-965d-6f1615d7db76
function cesar_cypher_text(texto, llave)
	cifrado = ""
	for letra in texto
		cifrado = string(cifrado, cesar_cypher_v2(letra, llave))
	end
	cifrado
end

# ╔═╡ 30ea543a-9552-4615-b4fe-f4a5f4200d85
cesar_cypher_text("HOLA", 3)

# ╔═╡ ede24985-989a-4513-b64b-99485d4d4720
md"""
Por último, solo para completar el problema, para descrifrar un mensaje basta con
desplazarlo en a dirección contraria de la llave. Es decir, si la llave es ``k``, 
hay que "cifrarlo" con la llave ``-k``
"""

# ╔═╡ 472fcb3b-52ac-4f8c-9f94-cb103c78061c
cesar_decypher_text(cifrado, llave) = cesar_cypher_text(cifrado, -llave)

# ╔═╡ 5107a398-72df-428d-927a-c9c482309df3
cesar_decypher_text(cesar_cypher_text("HOLA", 3), 3)

# ╔═╡ b7484cf8-a4ed-471e-a782-44626eb07c9f
md"""
## Referencias

* [vars](https://docs.julialang.org/en/v1/manual/variables/)
* [basic math](https://docs.julialang.org/en/v1/manual/mathematical-operations/)
* [string](https://docs.julialang.org/en/v1/manual/strings/)
* [functions](https://docs.julialang.org/en/v1/manual/functions/)
* [flow](https://docs.julialang.org/en/v1/manual/control-flow/)
* [data structures](https://docs.julialang.org/en/v1/base/collections/)
"""

# ╔═╡ Cell order:
# ╟─570ee268-868b-11eb-22cb-25d1a960cc58
# ╟─93c9a7e2-8762-11eb-1301-29befdbc7615
# ╟─08dc8a82-8776-11eb-3f1a-2b28aba9fd95
# ╠═45cb4852-8776-11eb-2291-c546e2fe7462
# ╠═492a3fbc-8776-11eb-3404-eb9c9b4343f1
# ╟─ebc8d1a2-8776-11eb-3c12-b1946911f134
# ╟─9761bfe8-8776-11eb-07ea-3f1569e3ad94
# ╠═d8e553d5-08c6-4cc1-90c1-a1b28b270752
# ╟─3e63a610-929e-4915-833d-c35318d35e5a
# ╠═c3a4b4f2-8776-11eb-3e07-410b48bb051b
# ╠═2296beea-8777-11eb-1482-938fe32607cd
# ╟─87c407f2-8777-11eb-2355-edd7cf9f535e
# ╟─352bd483-0561-4686-8979-e90b0c418a3f
# ╠═d84f9a56-9e86-4784-9b39-8246ad099973
# ╟─102190c4-8778-11eb-08a1-d3ba78f37a46
# ╠═b0cb8cd8-8777-11eb-0791-bb195b4e4cd4
# ╟─bb0151ba-8777-11eb-0028-8f9974bafaba
# ╠═f39f43ec-8777-11eb-23b0-db1d2d53b579
# ╟─fe25deb6-8777-11eb-395c-e7b6a72e5b74
# ╠═14200a8c-33da-43bc-840b-c9ba38cf3848
# ╟─efdac270-876f-11eb-0a14-c75fd8e64f57
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
# ╟─56fbf961-8e34-4838-8fbc-8dcf92251faf
# ╟─2ab0e75b-b07b-48d6-8dc4-413ab838af82
# ╠═8be85ec1-af78-4f88-870e-2d3950959265
# ╟─bae9fafe-2187-46c6-89f2-332c6f4eb689
# ╠═90e27d28-bb92-4318-af52-d95d79cd8c1f
# ╟─cc6cf9c2-87c1-40d6-8f70-920e143eeeb4
# ╟─e7ef2738-0d04-4726-8750-6d3af60e1c60
# ╠═0a794f1e-bae0-48bc-8094-c5700c029c19
# ╠═73841088-e2f4-4357-8eb4-e479a3da045b
# ╟─3e0501bf-45f9-4b04-a30c-ab844e40b24a
# ╠═2f56e4f3-f648-4742-aa97-df8dedec1650
# ╟─387b9b26-6e63-464b-865b-a8ba23c2e485
# ╟─42be3307-ebe3-4787-83c0-b12f30f70109
# ╠═ac03c2c4-b6f5-4d44-a424-72b12a9a05a0
# ╟─126d3903-6fad-4419-8d13-e0f2ebb98b91
# ╠═b483cfc0-e4db-4750-8999-f9d8e02ee797
# ╟─e25f1b55-cc54-4f53-8137-d4506a34a61c
# ╟─35a9372b-2a1c-44c3-b69a-f5ba60d3075d
# ╠═e854477e-61a9-4d2d-8252-9634e28beaa2
# ╠═3b21bc0f-e7f0-423f-8b80-92f48b4d35da
# ╟─2ec31d4c-f08e-4f5a-9f0a-d530350c8623
# ╠═a0c45f7d-00ef-45ef-ae2e-5527048985a2
# ╠═8e4cf0f3-84a7-4622-b2e4-064d6a0ac09b
# ╟─67fdce81-e5ab-412f-8ac9-99f6f16abae7
# ╟─da0124e3-8504-4f0c-a0bd-2d35f07fa34e
# ╠═b561a89e-72fe-4e06-88eb-dd4fdb7ea7d9
# ╟─daef06cb-b9d8-4b27-8f59-82745fd9ecb9
# ╠═e554f82c-8127-4c20-a6f0-458f24007532
# ╟─f61e722d-c34f-4ba4-98f4-d962792323ea
# ╟─f5153530-8d12-453a-86c8-f9a4e7803a7e
# ╠═0a379cce-43e3-4cfa-965d-6f1615d7db76
# ╠═30ea543a-9552-4615-b4fe-f4a5f4200d85
# ╟─ede24985-989a-4513-b64b-99485d4d4720
# ╠═472fcb3b-52ac-4f8c-9f94-cb103c78061c
# ╠═5107a398-72df-428d-927a-c9c482309df3
# ╟─b7484cf8-a4ed-471e-a782-44626eb07c9f
