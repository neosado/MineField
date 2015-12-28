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


bParallel = true
parallel = :local_

bAppend = false

if "parallel" in ARGS
    bParallel = true
elseif "serial" in ARGS
    bParallel = false
end

if "local" in ARGS
    parallel = :local_
elseif "remote" in ARGS
    parallel = :remote
elseif "both" in ARGS
    parallel = :both
end

if "append" in ARGS
    bAppend = true
end

if bParallel
    init_cluster(parallel)
    @everywhere include("simMineField.jl")
else
    include("simMineField.jl")
end

runExpBatch(bParallel = bParallel, bAppend = bAppend)


