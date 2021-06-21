### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 09f7d67e-cd66-40a3-939c-edb7804dcf5e
using DrWatson

# ╔═╡ 725bdb63-7047-47ef-bfa8-4951bdde19d0
begin
	using Images
	using FileIO
	using PlutoUI
end

# ╔═╡ 565da974-684e-4736-a40c-870dc1326046
@quickactivate "ss"

# ╔═╡ 01173cb7-ce50-4b35-805e-6be91663a320
md"""
# Manipulación de imágenes

El paquete `Images.jl` permite trabajar con imágenes como si fueran arreglos. Además,
incluye una gran cantidad de funciones para editarlas.
"""

# ╔═╡ 4416f363-c193-4693-b40c-25d945fec38d
md"""
La unidad de trabajo básica para las imágenes son los pixeles. Estos se representan
con los tipos `Gray`, `RBG`, `Lab`, dependiendo del tipo de imagen.
"""

# ╔═╡ 640b897b-b115-4c39-8d1e-661a15d2d7a5
@bind c Slider(0:0.1:1)

# ╔═╡ c7a21123-912c-44bc-bd6a-23ab2c288c88
gp = Gray(c)

# ╔═╡ 4b5e669d-00b4-4397-a8d3-8003bd2726b3
@bind r Slider(0:0.1:1)

# ╔═╡ cb9a19ef-c41b-4a9f-b5c1-ec145557399b
@bind g Slider(0:0.1:1)

# ╔═╡ 58eeb0c3-6e69-48dc-865f-d64b2031da0d
@bind b Slider(0:0.1:1)

# ╔═╡ 8cad8e1d-9c91-4a6b-a695-fa097e4c8ef5
pix = RGB(r, g, b)

# ╔═╡ a10d58e4-7cd3-4c94-b082-3f1cf3976963
md"""
Y las imágenes son arreglos de pixeles.
"""

# ╔═╡ b5d2b1fc-5495-48d7-ae97-c5eccbcf3768
grays = rand(3, 3)

# ╔═╡ 5db3c304-b8f5-4f04-b7cc-1bce8bdae6de
gray_im = Gray.(grays)

# ╔═╡ 5fb68200-78c5-4bff-ba9c-3f68390080bb
md"""
En el caso de los pixeles en blanco y negro, simplemente son un alias para números
decimales.
"""

# ╔═╡ 6d54b5c4-a37d-4c0b-b062-cf1b8ed4eeee
grays == gray_im

# ╔═╡ 0836d192-cb42-4a30-bc59-21c5fc0077d1
rgb_im = rand(RGB{Float32}, 3, 3)

# ╔═╡ Cell order:
# ╠═09f7d67e-cd66-40a3-939c-edb7804dcf5e
# ╠═565da974-684e-4736-a40c-870dc1326046
# ╠═725bdb63-7047-47ef-bfa8-4951bdde19d0
# ╟─01173cb7-ce50-4b35-805e-6be91663a320
# ╟─4416f363-c193-4693-b40c-25d945fec38d
# ╠═640b897b-b115-4c39-8d1e-661a15d2d7a5
# ╠═c7a21123-912c-44bc-bd6a-23ab2c288c88
# ╠═4b5e669d-00b4-4397-a8d3-8003bd2726b3
# ╠═cb9a19ef-c41b-4a9f-b5c1-ec145557399b
# ╠═58eeb0c3-6e69-48dc-865f-d64b2031da0d
# ╠═8cad8e1d-9c91-4a6b-a695-fa097e4c8ef5
# ╟─a10d58e4-7cd3-4c94-b082-3f1cf3976963
# ╠═b5d2b1fc-5495-48d7-ae97-c5eccbcf3768
# ╠═5db3c304-b8f5-4f04-b7cc-1bce8bdae6de
# ╟─5fb68200-78c5-4bff-ba9c-3f68390080bb
# ╠═6d54b5c4-a37d-4c0b-b062-cf1b8ed4eeee
# ╠═0836d192-cb42-4a30-bc59-21c5fc0077d1
