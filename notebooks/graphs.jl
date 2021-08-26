### A Pluto.jl notebook ###
# v0.15.0

using Markdown
using InteractiveUtils

# ╔═╡ 8fd8eb36-d62a-11eb-2ec9-8373932daaa5
using LightGraphs, GraphPlot

# ╔═╡ 16d33332-d38e-48e5-b325-9d27f8dcfd88
md"""
# Gráficas

En Julia hay varios paquetes para manipular gráficas. `LigthGraphs.jl` es uno de
ellos. Está orientado a la simplicidad y el alto rendimiento, además de ser bastante
modular.

Define un tipo abstracto `AbstractGraph` y una implementación optimizada `SimpleGraph`
y `SimpleDiGraph`, donde todos los nodos son enteros y no se pueden agregar atributos.
Hay varios paquetes que implementan gráficas con más caracterísiticas.

Las gráficas se pueden visualizar fácilmente tomando los vértices como puntos en el
plano y aristas como líneas que los unen. Hay varios paquetes que generan esta
visualización. Uno de los más sencillos es `GraphPlot`, que provee la función `gplot`.
"""

# ╔═╡ 2ca7c35b-e9ee-4a93-a4c7-b2eec31e6425
md"""
## Crea gráficas

Hay varias funciones para crear gráficas. Los constructores de los tipos son los más
sencillos. Permiten crear gráficas vacías con ``n`` vértices. Además, el tipo `Graph`
se usa como sinónimo de `SimpleGraph`.
"""

# ╔═╡ 19ffbf8f-817d-4aea-8de2-7d77ac562a59
gplot(Graph(5))

# ╔═╡ be3b06f6-7fc0-404d-b815-1cb3cedb8628
md"""
También se pueden crear gráficas a partir de su matriz de adyacencia.
"""

# ╔═╡ f6f051d2-a5c3-4ea8-a1a6-52ab2d494ebe
gplot(Graph([
		0 1 0;
		1 0 0;
		0 0 0
		]))

# ╔═╡ d2bf185b-825b-4b35-90ae-b7e7ef5988bc
gplot(SimpleDiGraph([
		0 1 0;
		1 0 0;
		0 1 0
		]))

# ╔═╡ 84ad4514-af45-4a35-bfb4-8f0eeb440dd7
md"""
Para gráficas más complicadas, hay funciones generadoras más específicas.
"""

# ╔═╡ a24ef434-24b4-4c8d-a2dc-0805f5cbc418
gplot(smallgraph(:karate))

# ╔═╡ e72b9b0d-a1de-4187-aa54-dc6f1bf3a500
gplot(clique_graph(3, 4))

# ╔═╡ 3f1ba432-cd90-435f-b991-7b9088e0452c
gplot(complete_bipartite_graph(2, 3), layout=circular_layout)

# ╔═╡ 36bdef24-8365-4ae4-b9f9-cef7c04621ac
md"""
## Propiedades de gráficas

`nv` da el número de vértices y `ne` el de aristas.
"""

# ╔═╡ ecccb5ad-0077-4f4a-bfb8-740b497713bb
g = Graph(5, 6)

# ╔═╡ 3cb91ae2-0168-41a2-925c-7e28e625ddd7
nv(g)

# ╔═╡ e4ef1fd7-33e1-4299-ae5c-e52bbf3e0489
ne(g)

# ╔═╡ a2a4db6b-1de2-42a2-bb5f-d2ba8ed3125f
gplot(g, nodelabel=1:nv(g), edgelabel=1:ne(g))

# ╔═╡ 7e78874e-0239-4b98-a9ee-2db095f29601
md"""
Se pueden iterar los vértices y aristas, además de obtener las matrices de adyacencia
e incidencia.
"""

# ╔═╡ c1d8ed79-569f-4fdd-86a4-fb4934d5b964
adjacency_matrix(g)

# ╔═╡ 5b7586f8-0d12-444d-aef6-a1f045d065e3
incidence_matrix(g)

# ╔═╡ 261c378c-2753-4b9d-97df-fc93a3d8ac72
md"""
## Operaciones básicas sobre gráficas

Se pueden añadir y quitar tanto vértices como aristas. Al ser solo enteros contiguos,
estas operaciones son simples.
"""

# ╔═╡ 09caca33-184b-4be5-94a7-646f840b90ca
add_vertex!(g)

# ╔═╡ 1ecb8527-7a44-49bf-b463-d4ae7b37d52f
gplot(g)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[compat]
GraphPlot = "~0.4.4"
LightGraphs = "~1.3.5"

[deps]
GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
LightGraphs = "093fc24a-ae57-5d10-9952-331d41423f4d"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "c6461fc7c35a4bb8d00905df7adafcff1fe3a6bc"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[GraphPlot]]
deps = ["ArnoldiMethod", "ColorTypes", "Colors", "Compose", "DelimitedFiles", "LightGraphs", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "dd8f15128a91b0079dfe3f4a4a1e190e54ac7164"
uuid = "a2cc645c-3eea-5389-862e-a155d0052231"
version = "0.4.4"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "daf7aec3fe3acb2131388f93a4c409b8c7f62226"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "745914ebcd610da69f3cb6bf76cb7bb83dcb8c9a"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.4"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═8fd8eb36-d62a-11eb-2ec9-8373932daaa5
# ╟─16d33332-d38e-48e5-b325-9d27f8dcfd88
# ╟─2ca7c35b-e9ee-4a93-a4c7-b2eec31e6425
# ╠═19ffbf8f-817d-4aea-8de2-7d77ac562a59
# ╟─be3b06f6-7fc0-404d-b815-1cb3cedb8628
# ╠═f6f051d2-a5c3-4ea8-a1a6-52ab2d494ebe
# ╠═d2bf185b-825b-4b35-90ae-b7e7ef5988bc
# ╟─84ad4514-af45-4a35-bfb4-8f0eeb440dd7
# ╠═a24ef434-24b4-4c8d-a2dc-0805f5cbc418
# ╠═e72b9b0d-a1de-4187-aa54-dc6f1bf3a500
# ╠═3f1ba432-cd90-435f-b991-7b9088e0452c
# ╟─36bdef24-8365-4ae4-b9f9-cef7c04621ac
# ╠═ecccb5ad-0077-4f4a-bfb8-740b497713bb
# ╠═3cb91ae2-0168-41a2-925c-7e28e625ddd7
# ╠═e4ef1fd7-33e1-4299-ae5c-e52bbf3e0489
# ╠═a2a4db6b-1de2-42a2-bb5f-d2ba8ed3125f
# ╟─7e78874e-0239-4b98-a9ee-2db095f29601
# ╠═c1d8ed79-569f-4fdd-86a4-fb4934d5b964
# ╠═5b7586f8-0d12-444d-aef6-a1f045d065e3
# ╟─261c378c-2753-4b9d-97df-fc93a3d8ac72
# ╠═09caca33-184b-4be5-94a7-646f840b90ca
# ╠═1ecb8527-7a44-49bf-b463-d4ae7b37d52f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
