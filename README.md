# Using GPU's from Julia

I am going to learn how to program GPUs, in particular Nvidia ones. From CUDA-C
to the higher level [wrappers](https://juliagpu.gitlab.io/CUDA.jl/) available in
Julia.

Then, I am going to select a problem and design a parallel solution to implement
it in Julia. At these point, I am considering the following problems.

* Optimization problems in grahps with a P underling solution

  Dijkstra, Bellman-Ford, Ford-Fulkersson.
* NP complete problems, to be solved with evolutionary techniques

  Particle Swarm Optimization, Ant Colony Optimization, Genetic Programming
