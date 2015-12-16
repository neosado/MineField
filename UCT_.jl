# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 10/15/2015

VERSION >= v"0.4" && __precompile__(false)


module UCT_

export UCT, selectAction, initialize, reinitialize


using MCTS_
using TreePolicyLib
using MDP_
using Util
using MCTSVisualizer_

using Base.Test


import MCTS_: selectAction, initialize, reinitialize


type TreePolicyParams

    bUCB1::Bool
    bUCB1_::Bool
    c::Float64

    bUCB1_tuned::Bool

    bUCB_V::Bool
    uv_c::Float64

    bUCBLike::Bool

    bTS::Bool

    bTSM::Bool
    arm_reward_model::Function

    bAUCB::Bool
    subpolicies::Vector{Dict{ASCIIString, Any}}
    control_policy::Dict{ASCIIString, Any}


    function TreePolicyParams(tree_policy::Any = nothing)

        self = new()

        self.bUCB1 = false
        self.bUCB1_tuned = false
        self.bUCB_V = false

        self.bUCBLike = false

        self.bUCB1_ = false
        self.bTS = false
        self.bTSM = false
        self.bAUCB = false

        if tree_policy == nothing
            self.bUCB1 = true
            self.bUCBLike = true
            self.c = sqrt(2)

        elseif tree_policy["type"] == :UCB1
            self.bUCB1 = true
            self.bUCBLike = true
            if haskey(tree_policy, "c")
                self.c = tree_policy["c"]
            else
                self.c = sqrt(2)
            end

        elseif tree_policy["type"] == :UCB1_
            self.bUCB1_ = true
            if haskey(tree_policy, "c")
                self.c = tree_policy["c"]
            else
                self.c = sqrt(2)
            end

        elseif tree_policy["type"] == :UCB1_tuned
            self.bUCB1_tuned = true
            self.bUCBLike = true
            if haskey(tree_policy, "c")
                self.ut_c = tree_policy["c"]
            else
                self.ut_c = 1/4
            end

        elseif tree_policy["type"] == :UCB_V
            self.bUCB_V = true
            self.bUCBLike = true
            self.uv_c = tree_policy["c"]

        elseif tree_policy["type"] == :TS
            self.bTS = true

        elseif tree_policy["type"] == :TSM
            self.bTSM = true
            self.arm_reward_model = tree_policy["ARM"]

        elseif tree_policy["type"] == :AUCB
            self.bAUCB = true
            self.subpolicies = tree_policy["SP"]
            if haskey(tree_policy, "CP")
                self.control_policy = tree_policy["CP"]
            else
                self.control_policy = Dict("type" => :TSN)
            end

        else
            error("Unknown tree policy type, ", tree_policy["type"])

        end

        return self
    end
end


type UCT <: MCTS

    seed::Union{Int64, Void}
    rng::AbstractRNG

    depth::Int64

    Generative::Function

    T::Dict{State, Bool}
    Ns::Dict{State, Int64}
    N::Dict{Tuple{State, Action}, Int64}
    Q::Dict{Tuple{State, Action}, Float64}

    X2::Dict{Tuple{State, Action}, Float64}

    TP::Dict{State, TreePolicy}

    nloop_max::Int64
    nloop_min::Int64
    eps::Float64

    runtime_max::Float64

    gamma_::Float64

    tree_policy::TreePolicyParams

    rollout_type::Symbol
    rollout_func::Function

    bReuse::Bool

    visualizer::Union{MCTSVisualizer, Void}


    function UCT(;seed::Union{Int64, Void} = nothing, depth::Int64 = 3, nloop_max::Int64 = 1000, nloop_min::Int64 = 100, eps::Float64 = 1e-3, runtime_max::Float64 = 0., gamma_::Float64 = 0.9, tree_policy::Any = nothing, rollout::Union{Tuple{Symbol, Function}, Void} = nothing, bReuse::Bool = true, visualizer::Union{MCTSVisualizer, Void} = nothing)

        self = new()

        if seed == nothing
            self.seed = round(Int64, time())
        else
            self.seed = seed
        end

        self.rng = MersenneTwister(self.seed)

        self.depth = depth

        self.Generative = Generative

        self.T = Dict{State, Bool}()
        self.Ns = Dict{State, Int64}()
        self.N = Dict{Tuple{State, Action}, Int64}()
        self.Q = Dict{Tuple{State, Action}, Float64}()

        self.X2 = Dict{Tuple{State, Action}, Float64}()

        self.TP = Dict{State, TreePolicy}()

        self.nloop_max = nloop_max
        self.nloop_min = nloop_min
        self.eps = eps

        self.runtime_max = runtime_max

        self.gamma_ = gamma_

        self.tree_policy = TreePolicyParams(tree_policy)

        if rollout == nothing
            self.rollout_type = :default
            self.rollout_func = rollout_default
        else
            self.rollout_type = rollout[1]
            self.rollout_func = rollout[2]
        end

        self.bReuse = bReuse

        self.visualizer = visualizer

        return self
    end
end


function Generative(pm::MDP, s::State, a::Action)

    s_, r = MDP_.Generative(pm, s, a)

    return s_, r / pm.reward_norm_const
end


function default_policy(pm::MDP, s::State)
    
    # Note: pass pm.rng to rand() if pm supports rng

    a = pm.actions[rand(1:pm.nActions)]

    while !isFeasible(pm, s, a)
        a = pm.actions[rand(1:pm.nActions)]
    end

    return a
end


function rollout_default(alg::UCT, pm::MDP, s::State, d::Int64; rgamma::Float64 = 0.9, debug::Int64 = 0)

    if d == 0 || isEnd(pm, s)
        return 0
    end

    a = default_policy(pm, s)

    if debug > 2
        print(string(a), ", ")
    end

    s_, r = alg.Generative(pm, s, a)

    return r + rgamma * rollout_default(alg, pm, s_, d - 1, rgamma = rgamma, debug = debug)
end


function simulate(alg::UCT, pm::MDP, s::State, d::Int64; debug::Int64 = 0)

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :start_sim, s)
    end

    bEnd = isEnd(pm, s)

    if d == 0 || bEnd
        if !haskey(alg.T, s)
            if !bEnd
                for a in pm.actions
                    alg.N[(s, a)] = 0

                    if isFeasible(pm, s, a)
                        alg.Q[(s, a)] = 0
                    else
                        alg.Q[(s, a)] = -Inf
                    end

                    alg.X2[(s, a)] = 0
                end

                if !alg.tree_policy.bUCBLike
                    feasible_actions = Dict{Action, Bool}()

                    for a in pm.actions
                        if isFeasible(pm, s, a)
                            feasible_actions[a] = true
                        else
                            feasible_actions[a] = false
                        end
                    end

                    if alg.tree_policy.bUCB1_
                        alg.TP[s] = UCB1Policy(pm, feasible_actions, c = alg.tree_policy.c)
                    elseif alg.tree_policy.bTS
                        alg.TP[s] = TSPolicy(pm, feasible_actions)
                    elseif alg.tree_policy.bTSM
                        alg.TP[s] = TSMPolicy(pm, feasible_actions, alg.tree_policy.arm_reward_model)
                    elseif alg.tree_policy.bAUCB
                        alg.TP[s] = AUCBPolicy(pm, feasible_actions, alg.tree_policy.subpolicies, control_policy = alg.tree_policy.control_policy)
                    end
                end
            end

            alg.T[s] = true
            alg.Ns[s] = 0
        end

        if bEnd && debug > 2
            println("    hit end")
        end

        return 0
    end

    if !haskey(alg.T, s)
        if debug > 2
            println("    new node: ", s, " at level ", d)
        end

        for a in pm.actions
            alg.N[(s, a)] = 0

            if isFeasible(pm, s, a)
                alg.Q[(s, a)] = 0
            else
                alg.Q[(s, a)] = -Inf
            end

            alg.X2[(s, a)] = 0
        end

        if !alg.tree_policy.bUCBLike
            feasible_actions = Dict{Action, Bool}()

            for a in pm.actions
                if isFeasible(pm, s, a)
                    feasible_actions[a] = true
                else
                    feasible_actions[a] = false
                end
            end

            if alg.tree_policy.bUCB1_
                alg.TP[s] = UCB1Policy(pm, feasible_actions, c = alg.tree_policy.c)
            elseif alg.tree_policy.bTS
                alg.TP[s] = TSPolicy(pm, feasible_actions)
            elseif alg.tree_policy.bTSM
                alg.TP[s] = TSMPolicy(pm, feasible_actions, alg.tree_policy.arm_reward_model)
            elseif alg.tree_policy.bAUCB
                alg.TP[s] = AUCBPolicy(pm, feasible_actions, alg.tree_policy.subpolicies, control_policy = alg.tree_policy.control_policy)
            end
        end

        alg.T[s] = true
        alg.Ns[s] = 0

        ro = alg.rollout_func(alg, pm, s, d, debug = debug)

        if debug > 2
            println("    rollout: ", neat(ro * pm.reward_norm_const))
        end

        return ro
    end

    if debug > 2
        println("    found node: ", s, " at level ", d)
    end

    Qv = Array(Float64, pm.nActions)

    if alg.tree_policy.bUCBLike
        if alg.tree_policy.bUCB1_tuned || alg.tree_policy.bUCB_V
            var_ = zeros(pm.nActions)
            RE = zeros(pm.nActions)
        end

        for i = 1:pm.nActions
            a = pm.actions[i]

            if !isFeasible(pm, s, a)
                Qv[i] = -Inf

            elseif alg.N[(s, a)] == 0
                Qv[i] = Inf

            else
                if alg.tree_policy.bUCB1
                    Qv[i] = alg.Q[(s, a)] + alg.tree_policy.c * sqrt(log(alg.Ns[s]) / alg.N[(s, a)])

                elseif alg.tree_policy.bUCB1_tuned || alg.tree_policy.bUCB_V
                    if alg.N[(s, a)] > 1
                        var_[i] = (alg.X2[(s, a)] - alg.N[(s, a)] * (alg.Q[(s, a)] * alg.Q[(s, a)])) / (alg.N[(s, a)] - 1)
                        if abs(var_[i]) < 1.e-7
                            var_[i] = 0.
                        end
                        @assert var_[i] >= 0.
                        RE[i] = sqrt(var_[i]) / abs(alg.Q[(s, a)])
                    end

                    if alg.tree_policy.bUCB1_tuned
                        Qv[i] = alg.Q[(s, a)] + sqrt(log(alg.Ns[s]) / alg.N[(s, a)] * min(alg.tree_policy.ut_c, var_[i] + sqrt(2 * log(alg.Ns[s]) / alg.N[(s, a)])))

                    elseif alg.tree_policy.bUCB_V
                        Qv[i] = alg.Q[(s, a)] + sqrt(2 * var_[i] * log(alg.Ns[s]) / alg.N[(s, a)]) + alg.tree_policy.uv_c * 3 * log(alg.Ns[s]) / alg.N[(s, a)]

                    end

                end

            end
        end

        a = pm.actions[argmax(Qv)]

    else
        if alg.tree_policy.bAUCB
            a, Qv, sindex = TreePolicyLib.selectAction(alg.TP[s], pm)
        else
            a, Qv = TreePolicyLib.selectAction(alg.TP[s], pm)
        end

    end

    s_, r = Generative(pm, s, a)

    if debug > 2
        println("    Q: ", neat(Q * pm.reward_norm_const), ", Qv: ", neat(Qv), ", a: ", a, ", s_: ", s_, ", r: ", neat(r * pm.reward_norm_const))
        if debug > 3
            Na = zeros(Int64, pm.nActions)
            for i = 1:pm.nActions
                Na[i] = alg.N[(s, pm.actions[i])]
            end
            println("    Ns: ", alg.Ns[s], ", N: ", Na)
        end
        if alg.tree_policy.bUCB1_tuned || alg.tree_policy.bUCB_V
            println("    var: ", neat(var_), ", RE: ", neat(RE))
        end
    end

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :before_sim, s, a)
    end

    q = r + alg.gamma_ * simulate(alg, pm, s_, d - 1, debug = debug)

    alg.Ns[s] += 1
    alg.N[(s, a)] += 1
    alg.Q[(s, a)] += (q - alg.Q[(s, a)]) / alg.N[(s, a)]
    alg.X2[(s, a)] += q * q

    if !alg.tree_policy.bUCBLike
        if alg.tree_policy.bAUCB
            updatePolicy(alg.TP[s], pm, a, q, sindex)
        else
            updatePolicy(alg.TP[s], pm, a, q)
        end
    end

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :after_sim, s, a, r * pm.reward_norm_const, q * pm.reward_norm_const, alg.Ns[s], alg.N[(s, a)], alg.Q[(s, a)] * pm.reward_norm_const)
    end

    return q
end


function selectAction(alg::UCT, pm::MDP, s::State; debug::Int64 = 0)

    if alg.visualizer != nothing
        initTree(alg.visualizer)
    end

    if !alg.bReuse
        initialize(alg)
    end

    Q = Dict{Action, Float64}()
    for a in pm.actions
        Q[a] = 0.
    end

    if debug > 2
        println("  state: ", s)
    end

    start_time = time()

    n = 0
    res = 0.

    for i = 1:alg.nloop_max
        if debug > 2
            println("  iteration: ", i)
        end

        n = i

        simulate(alg, pm, s, alg.depth, debug = debug)

        #println("h: ", h)
        #println("T: ", alg.T)
        #println("Ns: ", alg.Ns)
        #println("N: ", alg.N)
        #println("Q: ", alg.Q)
        #println()

        res = 0.

        for a in pm.actions
            Q_prev = Q[a]
            Q[a] = alg.Q[(s, a)] * pm.reward_norm_const
            if !isinf(Q[a])
                res += (Q[a] - Q_prev)^2
            end
        end

        if i >= alg.nloop_min
            if sqrt(res) < alg.eps
                break
            elseif alg.runtime_max != 0 && time() - start_time > alg.runtime_max
                break
            end
        end
    end

    if debug > 1
        println("  # of iterations: ", n)
        println("  residual: ", sqrt(res))
    end

    Q_max = -Inf
    for a in pm.actions
        Q[a] = alg.Q[(s, a)] * pm.reward_norm_const

        if Q[a] > Q_max
            Q_max = Q[a]
        end
    end

    actions = Action[]
    for a in pm.actions
        if Q[a] == Q_max
            push!(actions, a)
        end
    end
    action = actions[randi(alg.rng, 1:length(actions))]

    if alg.visualizer != nothing
        saveTree(alg.visualizer, pm)
    end

    return action, Q
end


function initialize(alg::UCT)

    alg.T = Dict{State, Bool}()
    alg.Ns = Dict{State, Int64}()
    alg.N = Dict{Tuple{State, Action}, Int64}()
    alg.Q = Dict{Tuple{State, Action}, Float64}()

    alg.X2 = Dict{Tuple{State, Action}, Float64}()

    alg.TP= Dict{State, TreePolicy}()
end


reinitialize(alg::UCT) = error("$(typeof(alg)) does not implement reinitialize()")


end


