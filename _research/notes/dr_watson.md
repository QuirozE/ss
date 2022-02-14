# Dr. Watson peculiarities

## Projects are not packages

For some god forsaken reason, in Julia some automated functionalities (build
process, units tests) are restricted to packages.

See [this](https://github.com/JuliaDynamics/DrWatson.jl/issues/62) and
[this](https://github.com/JuliaLang/Pkg.jl/pull/1215).

## Unit tests

Projects can't have tests. To create tests, create a package inside `src`, wich
can actually have tests.

See [this](https://github.com/JuliaDynamics/DrWatson.jl/issues/67) and
[this](https://github.com/JuliaDynamics/DrWatson.jl/issues/130).
