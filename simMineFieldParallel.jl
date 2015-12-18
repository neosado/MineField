# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 12/14/2015

function init_cluster(parallel::Symbol = :local_)

    ncpu_local = CPU_CORES / 2
    machines = [
        ("youngjun@tula", 20, "/home/youngjun/MineField"),
        ("youngjun@cheonan", 4, "/home/youngjun/MineField"),
        ("youngjun@cambridge", 6, "/home/youngjun/MineField")
    ]

    if parallel == :local_ || parallel == :both
        addprocs(round(Int64, ncpu_local))
    end

    if parallel == :remote || parallel == :both
        for (machine, count, dir) in machines
            cluster_list = ASCIIString[]

            for i = 1:count
                push!(cluster_list, machine)
            end

            addprocs(cluster_list, dir = dir)
        end
    end
end


init_cluster()
@everywhere include("simMineField.jl")

runExpBatch(bParallel = true)


