# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 10/15/2015

push!(LOAD_PATH, ".")

using MDP_
using MineField_

using UCT_

using MineFieldVisualizer_
using MCTSVisualizer_

using Dist_
using Util

using Distributions
using Base.Collections


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


function simulate(pm, alg; draw::Bool = true, wait::Bool = false, debug::Int64 = 0)

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

    if draw
        visInit(mfv, pm)
        visUpdate(mfv, pm, ts, s)
        updateAnimation(mfv, ts)
    end

    actions = Array(Action, pm.nx + pm.ny - 2)

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

            println("time: ", ts, ", s: ", string(s), ", Q: ", neat(Q__), ", a: ", string(a), ", r: ", neat(r), ", R: ", neat(R), ", s_: ", string(s_))
        end

        if draw
            visInit(mfv, pm)
            visUpdate(mfv, pm, ts, s_, a, neat(r), neat(R))
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

    return R, actions
end


#
# Driver
#

function generateRewardMap(nx::Int64, ny::Int64; seed::Union{Int64, Void} = nothing)

    if seed != nothing
        if seed != 0
            srand(seed)
        else
            srand(round(Int64, time()))
        end
    end

    Reward = Array(RareDist, nx, ny)

    for i = 1:nx
        for j = 1:ny
            p = rand() / 5
            mu = -10 - 40 * rand()
            sigma = 1 + 4 * rand()

            Reward[i, j] = RareDist(p, -1000., Truncated(Normal(mu, sigma), -Inf, -1))
        end
    end

    Reward[1, 1]= RareDist(1., 0., Normal())
    Reward[nx, ny] = RareDist(1., 0., Normal())

    return Reward
end


nx = 7
ny = 5

reward_seed = 23
mf_seed = round(Int64, time())
mcts_seed = round(Int64, time())


println("nx: ", nx, ", ny: ", ny)
println("seed: ", reward_seed, ", ", mf_seed, ", ", mcts_seed)
println()

pm = MineField(nx, ny, seed = mf_seed, Reward = generateRewardMap(nx, ny, seed = reward_seed))

RM = computeRewardMap(pm)
println("Reward Map: ")
println(neat(rotl90(RM)))

distance, path = computeShortestPath(-RM, pm.rover_init_loc, pm.destination)

println("Optimal Path: ", path)
println("Maximum Return: ", neat(-distance))
println()

#alg = UCT(depth = 5, nloop_max = 10000, nloop_min = 1000, visualizer = MCTSVisualizer())
alg = UCT(seed = mcts_seed, depth = nx + ny - 2, nloop_max = 1000000, nloop_min = 100)

#test(pm, alg)

R, actions = simulate(pm, alg, draw = false, wait = false, debug = 2)

actions_ = Array(Symbol, nx + ny - 2)
path = Array(Tuple{Int64, Int64}, nx + ny - 1)
expected_return = 0

path[1] = (1, 1)
for i = 1:(nx + ny - 2)
    actions_[i] = actions[i].action

    if actions_[i] == :up
        path[i + 1] = (path[i][1], path[i][2] + 1)
    elseif actions_[i] == :right
        path[i + 1] = (path[i][1] + 1, path[i][2])
    end

    expected_return += RM[path[i + 1]...]
end

println("Actions: ", actions_)
println("Path: ", path)
println("Expected Return: ", neat(expected_return))
println("Actual Return: ", neat(R))


