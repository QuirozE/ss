# There are some CUDA and C++ programs that produce an output to render. I am
# not experienced enough in those languages to do a robust integration with some
# visualization library, so I just store the results as CSV files, and use this
# script to create nice plots with Plots.jl

using DrWatson
@quickactivate("ss")

using DelimitedFiles
using Plots; pgfplotsx()

function graph_csv(filename)
    (points, _) = readdlm(
        datadir("sims", filename),
        ',',
        Float64,
        header=true
    )

    scatter(
        points[:, 1], points[:, 2],
        label="",
        markersize=0.1,
        markerstrokewidth=0,
    )
    savefig(plotsdir(filename * ".tikz"))
end

function parse_and_graph()
    if length(ARGS) < 1
        print("Filename is required")
        return
    end
    graph_csv(ARGS[1])
end

parse_and_graph()
