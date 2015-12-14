using JSON


function init_cluster(parallel::Symbol = :local_)

    ncpu_local = CPU_CORES / 2
    machines = [("youngjun@tula", 20, "/usr/bin"), ("youngjun@cheonan", 4, "/usr/bin"), ("youngjun@cambridge", 6, "/usr/bin")]

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


function runExpParallel(bParallel::Bool = false)

    N = 100

    nx = 7
    ny = 5

    srand(12)

    reward_seed = rand(10000:typemax(Int16))
    mf_seed_list = unique(rand(10000:typemax(Int16), round(Int64, N * 1.1)))[1:N]
    mcts_seed_list = unique(rand(10000:typemax(Int16), round(Int64, N * 1.1)))[1:N]

    tree_policies = Dict{ASCIIString,Any}[Dict("type" => :UCB1, "c" => 1), Dict("type" => :UCB1, "c" => 300)]

    D = Dict{Dict{ASCIIString,Any},Dict{ASCIIString,Any}}()

    if bParallel
        if true
            for tree_policy in tree_policies
                results = pmap(id -> runExp(reward_seed, mf_seed_list[id], mcts_seed_list[id], tree_policy, 1, nx = nx, ny = ny, nloop_max = 1000000, nloop_min = 100, bParallel = true, id = id), 1:N)

                opt_dist = 0
                expected_returns = zeros(N)

                for result in results
                    id = result[1]
                    opt_dist = result[2]
                    expected_returns[id] = result[3]
                end

                D[tree_policy] = Dict("reward_seed" => reward_seed, "mf_seed_list" => mf_seed_list, "mcts_seed_list" => mcts_seed_list, "N" => N, "nx" => nx, "ny" => ny, "opt_dist" => opt_dist, "expected_returns" => expected_returns)
            end

        else
            results = pmap(tree_policy -> runExp(reward_seed, mf_seed_list, mcts_seed_list, tree_policy, N, nx = nx, ny = ny, nloop_max = 1000000, nloop_min = 100, bParallel = true, id = tree_policy), tree_policies)

            for result in results
                tree_policy = result[1]
                opt_dist = result[2]
                expected_returns = result[3]

                D[tree_policy] = Dict("reward_seed" => reward_seed, "mf_seed_list" => mf_seed_list, "mcts_seed_list" => mcts_seed_list, "N" => N, "nx" => nx, "ny" => ny, "opt_dist" => opt_dist, "expected_returns" => expected_returns)
            end

        end

    else
        for tree_policy in tree_policies
            opt_dist, expected_returns = runExp(reward_seed, mf_seed_list, mcts_seed_list, tree_policy, N, nx = nx, ny = ny, nloop_max = 1000000, nloop_min = 100)
            D[tree_policy] = Dict("reward_seed" => reward_seed, "mf_seed_list" => mf_seed_list, "mcts_seed_list" => mcts_seed_list, "N" => N, "nx" => nx, "ny" => ny, "opt_dist" => opt_dist, "expected_returns" => expected_returns)
        end

    end

    f = open("exp.json", "w")
    JSON.print(f, D)
    close(f)
end


bParallel = true

if bParallel
    init_cluster()
    @everywhere include("simMineField.jl")
end

runExpParallel(bParallel)


