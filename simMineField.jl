# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 10/15/2015

push!(LOAD_PATH, ".")

using MDP_
using MineField_

using UCT_

using MineFieldVisualizer_
using MCTSVisualizer_

using Dist_
using ArmRewardModel_
using Util

using Distributions
using Base.Collections
using JLD


#
# Dijkstra's Algorithm
#

function getAdj(nx::Int64, ny::Int64, u::Tuple{Int64, Int64})

    Adj = Tuple{Int64, Int64}[]

    if u[1] != 1
        push!(Adj, (u[1] - 1, u[2]))
    end

    if u[1] != nx
        push!(Adj, (u[1] + 1, u[2]))
    end

    if u[2] != 1
        push!(Adj, (u[1], u[2] - 1))
    end

    if u[2] != ny
        push!(Adj, (u[1], u[2] + 1))
    end

    return Adj
end

# Based on Dijkstra's shortest path algorithm
function computeShortestPath(W::Array{Float64, 2}, s::Tuple{Int64, Int64}, t::Tuple{Int64, Int64}; debug::Int64 = 0)

    if debug > 0
        println(W)
    end

    nx, ny = size(W)

    OptDist = fill(Inf, size(W))
    OptPath = fill((0, 0), size(W))
    OptDist[s...] = 0

    if debug > 0
        println(OptDist)
        println(OptPath)
    end

    S = Set()

    Q = PriorityQueue([(i, j) => OptDist[i, j] for i = 1:nx, j = 1:ny]);

    if debug > 0
        println(Q)
        println()
    end

    while !isempty(Q)
        u = dequeue!(Q)
        union!(S, [u])

        for v in getAdj(nx, ny, u)
            if OptDist[v...] > OptDist[u...] + W[v...]
                OptDist[v...] = OptDist[u...] + W[v...]
                OptPath[v...] = u
                Q[v...] = OptDist[v...]
            end
        end

        if debug > 0
            println(Q)
            println(u)
            println(S)
            println(OptDist)
            println(OptPath)

            readline()
        end
    end

    path = Tuple{Int64, Int64}[]
    v = t
    while v != s
        prepend!(path, [v])
        v = OptPath[v...]
    end
    prepend!(path, [s])

    return OptDist[t...], path
end


#
# UCT
#

function getInitialState(pm::MineField)

    return MFState(pm.rover_init_loc...)
end


function test(pm, alg)

    s = getInitialState(pm)

    a_opt, Q = selectAction(alg, pm, s)

    Q__ = Float64[]
    for a in  pm.actions
        push!(Q__, Q[a])
    end

    println("Q: ", neat(Q__))
    println("action: ", a_opt.action)
end


function simulate(pm, alg; draw::Bool = false, wait::Bool = false, debug::Int64 = 0)

    if draw
        mfv = MineFieldVisualizer(wait = wait)
    end

    ts = 0

    s = getInitialState(pm)

    R = 0

    if draw
        visInit(mfv, pm)
        visUpdate(mfv, pm)
        updateAnimation(mfv)
    end

    if debug > 0
        println("time: 0, s: ", string(s))
    end

    expected_reward_map = computeExpectedRewardMap(pm.Reward)

    if draw
        visInit(mfv, pm, expected_reward_map)
        visUpdate(mfv, pm, ts, s)
        updateAnimation(mfv, ts)
    end

    actions = Array(Action, pm.nx + pm.ny - 2)
    path = Tuple{Int64, Int64}[]
    push!(path, (s.x, s.y))
    expected_return = 0

    while true
        ts += 1

        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println()

        a, Q = selectAction(alg, pm, s, debug = debug)

        actions[ts] = a

        #println("T: ", alg.T)
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
        #println("Q: ", alg.Q)
        #println()

        s_, r = Generative(pm, s, a)

        push!(path, (s_.x, s_.y))
        expected_return += expected_reward_map[s_.x, s_.y]

        R += r

        if debug > 0
            if debug > 2
                for a__ in pm.actions
                    println(string(a__), ": ", alg.N[(s, a__)]) 
                end
            end

            Q__ = Float64[]
            for a__ in  pm.actions
                push!(Q__, Q[a__])
            end

            println("time: ", ts, ", s: ", string(s), ", Q: ", neat(Q__), ", a: ", string(a), ", r: ", neat(r), ", R: ", neat(R), ", ER: ", neat(expected_return), ", s_: ", string(s_))
        end

        if draw
            visInit(mfv, pm, expected_reward_map, path)
            visUpdate(mfv, pm, ts, s_, a, neat(r), neat(R), neat(expected_return))
            updateAnimation(mfv, ts)
        end

        s = s_

        if isEnd(pm, s)
            if debug > 0
                println("reached the terminal state")
            end

            break
        end
    end

    if draw
        saveAnimation(mfv, repeat = true)
    end

    return R, actions, path, expected_return
end


#
# Driver
#

function generateRewardMap(nx::Int64, ny::Int64; seed::Union{Int64, Void} = nothing)

    if seed == nothing
        seed = round(Int64, time())
    end

    rng = MersenneTwister(seed)

    Reward = Array(Union{RareDist, FixedValue}, nx, ny)

    for i = 1:nx
        for j = 1:ny
            p = rand(rng) / 5  # 0. ~ 0.2
            mu = -10 - 40 * rand(rng)
            sigma = 1 + 4 * rand(rng)

            Reward[i, j] = RareDist(p, -1000., Truncated(Normal(mu, sigma), mu - 5 * sigma, min(mu + 5 * sigma, -1.)))
        end
    end

    Reward[1, 1]= FixedValue(0.)
    Reward[nx, ny] = FixedValue(0.)

    return Reward
end


function runExp(reward_seed::Int64, mf_seed::Union{Int64, Vector{Int64}}, mcts_seed::Union{Int64, Vector{Int64}}, tree_policy::Any, N::Int64; nx::Int64 = 5, ny::Int64 = 5, nloop_min::Int64 = 100, nloop_max::Int64 = 1000, bParallel::Bool = false, id::Any = nothing)

    @assert length(mf_seed) == N
    @assert length(mcts_seed) == N

    expected_returns = zeros(N)

    Reward = generateRewardMap(nx, ny, seed = reward_seed)

    expected_reward_map = computeExpectedRewardMap(Reward)
    distance, path = computeShortestPath(-expected_reward_map, (1, 1), (nx, ny))

    for i = 1:N
        pm = MineField(nx, ny, seed = mf_seed[i], Reward = Reward)

        alg = UCT(seed = mcts_seed[i], depth = nx + ny - 2, nloop_max = nloop_max, nloop_min = nloop_min, tree_policy = tree_policy)

        R, actions, path, expected_return = simulate(pm, alg)

        expected_returns[i] = expected_return
    end

    if N == 1
        expected_returns = expected_returns[1]
    end

    if bParallel
        return id, -distance, expected_returns
    else
        return -distance, expected_returns
    end
end


function runExpBatch(; bParallel::Bool = false)

    nx = 7
    ny = 5

    srand(12)
    nScenarios = 30

    tree_policies = Dict{ASCIIString, Any}[Dict("type" => :UCB1, "c" => 1.), Dict("type" => :UCB1, "c" => 300.), Dict("type" => :TS), Dict("type" => :TSM, "ARM" => () -> ArmRewardModel(0.01, 0.01, -50., 1., 1 / 2, 1 / (2 * (1 / abs(5) ^ 2)), -500., -1000., 1., 1 / 2, 0.001)), Dict("type" => :AUCB, "SP" => [Dict("type" => :UCB1, "c" => 1.), Dict("type" => :UCB1, "c" => 300.)])]

    N = 100


    scenarios = unique(rand(10000:typemax(Int16), round(Int64, nScenarios * 1.1)))[1:nScenarios]

    for scenario in scenarios
        println("Scenario: ", scenario)

        srand(scenario)

        mf_seed_list = unique(rand(10000:typemax(Int16), round(Int64, N * 1.1)))[1:N]
        mcts_seed_list = unique(rand(10000:typemax(Int16), round(Int64, N * 1.1)))[1:N]

        D = Dict{Tuple{Int64, Dict{ASCIIString, Any}}, Dict{ASCIIString, Any}}()

        if bParallel
            if true
                for tree_policy in tree_policies
                    results = pmap(id -> runExp(scenario, mf_seed_list[id], mcts_seed_list[id], tree_policy, 1, nx = nx, ny = ny, nloop_max = 1000000, nloop_min = 100, bParallel = true, id = id), 1:N)

                    opt_dist = 0
                    expected_returns = zeros(N)

                    for result in results
                        id = result[1]
                        opt_dist = result[2]
                        expected_returns[id] = result[3]
                    end

                    D[(scenario, tree_policy)] = Dict("mf_seed_list" => mf_seed_list, "mcts_seed_list" => mcts_seed_list, "N" => N, "nx" => nx, "ny" => ny, "opt_dist" => opt_dist, "expected_returns" => expected_returns)
                end

            else
                results = pmap(tree_policy -> runExp(scenario, mf_seed_list, mcts_seed_list, tree_policy, N, nx = nx, ny = ny, nloop_max = 1000000, nloop_min = 100, bParallel = true, id = tree_policy), tree_policies)

                for result in results
                    tree_policy = result[1]
                    opt_dist = result[2]
                    expected_returns = result[3]

                    D[(scenario, tree_policy)] = Dict("mf_seed_list" => mf_seed_list, "mcts_seed_list" => mcts_seed_list, "N" => N, "nx" => nx, "ny" => ny, "opt_dist" => opt_dist, "expected_returns" => expected_returns)
                end

            end

        else
            for tree_policy in tree_policies
                opt_dist, expected_returns = runExp(scenario, mf_seed_list, mcts_seed_list, tree_policy, N, nx = nx, ny = ny, nloop_max = 1000000, nloop_min = 100)
                D[(scenario, tree_policy)] = Dict("mf_seed_list" => mf_seed_list, "mcts_seed_list" => mcts_seed_list, "N" => N, "nx" => nx, "ny" => ny, "opt_dist" => opt_dist, "expected_returns" => expected_returns)
            end

        end

        if isfile("exp.jld")
            D_ = load("exp.jld", "DATA");

            for (key, experiment) in D
                scenario, tree_policy = key
                D_[(scenario, tree_policy)] = experiment
            end

        else
            D_ = D

        end

        save("exp.jld", "DATA", D_)
    end
end


if false
    nx = 7
    ny = 5

    reward_seed = 23
    mf_seed = round(Int64, time())
    mcts_seed = round(Int64, time())


    println("nx: ", nx, ", ny: ", ny)
    println("seed: ", reward_seed, ", ", mf_seed, ", ", mcts_seed)
    println()

    Reward = generateRewardMap(nx, ny, seed = reward_seed)

    pm = MineField(nx, ny, seed = mf_seed, Reward = Reward)

    expected_reward_map = computeExpectedRewardMap(Reward)

    println("Reward Map: ")
    println(neat(rotl90(expected_reward_map)))

    distance, path = computeShortestPath(-expected_reward_map, pm.rover_init_loc, pm.destination)

    println("Optimal Path: ", path)
    println("Maximum Return: ", neat(-distance))
    println()

    tree_policy = Dict("type" => :UCB1, "c" => 300.)
    #tree_policy = Dict("type" => :UCB1_, "c" => 300.)
    #tree_policy = Dict("type" => :TS)
    #tree_policy = Dict("type" => :TSM, "ARM" => () -> ArmRewardModel(0.01, 0.01, -50., 1., 1 / 2, 1 / (2 * (1 / abs(5) ^ 2)), -500., -1000., 1., 1 / 2, 0.001))
    #tree_policy = Dict("type" => :AUCB, "SP" => [Dict("type" => :UCB1, "c" => 1.), Dict("type" => :UCB1, "c" => 300.)])

    alg = UCT(seed = mcts_seed, depth = nx + ny - 2, nloop_max = 1000000, nloop_min = 100, tree_policy = tree_policy, visualizer = MCTSVisualizer())

    #test(pm, alg)

    R, actions, path, expected_return = simulate(pm, alg, draw = true, wait = true, debug = 2)

    actions_ = Array(Symbol, nx + ny - 2)
    for i = 1:(nx + ny - 2)
        actions_[i] = actions[i].action
    end

    println("Actions: ", actions_)
    println("Path: ", path)
    println("Expected Return: ", neat(expected_return))
    println("Actual Return: ", neat(R))
end


