# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 10/15/2015

module UCT_

export UCT, selectAction, initialize, reinitialize


using MCTS_
using MDP_
using Util
using MCTSVisualizer_

using Base.Test


import MCTS_: selectAction, initialize, reinitialize


type TreePolicy

    bUCB1::Bool
    c::Float64

    bUCB1_tuned::Bool

    bUCB_V::Bool
    uv_c::Float64


    function TreePolicy(tree_policy::Any = nothing)

        self = new()

        self.bUCB1 = false
        self.bUCB1_tuned = false
        self.bUCB_V = false

        if tree_policy == nothing
            self.bUCB1 = true
            self.c = sqrt(2)

        else
            if typeof(tree_policy) <: Dict
                tree_policy = Any[tree_policy]
            end

            for tree_policy_ in tree_policy
                if tree_policy_["type"] == :UCB1
                    self.bUCB1 = true
                    self.c = tree_policy_["c"]

                elseif tree_policy_["type"] == :UCB1_tuned
                    self.bUCB1_tuned = true

                elseif tree_policy_["type"] == :UCB_V
                    self.bUCB_V = true
                    self.uv_c = tree_policy_["c"]

                end
            end

        end

        return self
    end
end


type UCT <: MCTS

    seed::Union{Int64, Void}

    depth::Int64

    Generative::Function

    T::Dict{State, Bool}
    N::Dict{Tuple{State, Action}, Int64}
    Ns::Dict{State, Int64}
    Q::Dict{Tuple{State, Action}, Float64}

    X2::Dict{Tuple{State, Action}, Float64}

    nloop_max::Int64
    nloop_min::Int64
    eps::Float64

    runtime_max::Float64

    gamma_::Float64

    tree_policy::TreePolicy

    rollout_type::Symbol
    rollout_func::Function

    bReuse::Bool

    visualizer::Union{MCTSVisualizer, Void}


    function UCT(;seed::Union{Int64, Void} = nothing, depth::Int64 = 3, nloop_max::Int64 = 1000, nloop_min::Int64 = 100, eps::Float64 = 1e-3, runtime_max::Float64 = 0., gamma_::Float64 = 0.9, tree_policy::Any = nothing, rollout::Union{Tuple{Symbol, Function}, Void} = nothing, bReuse::Bool = true, visualizer::Union{MCTSVisualizer, Void} = nothing)

        self = new()

        if seed != nothing
            if seed != 0
                self.seed = seed
            else
                self.seed = round(Int64, time())
            end

            srand(self.seed)

        else
            self.seed = nothing

        end

        self.depth = depth

        self.Generative = Generative

        self.T = Dict{State, Bool}()
        self.N = Dict{Tuple{State, Action}, Int64}()
        self.Ns = Dict{State, Int64}()
        self.Q = Dict{Tuple{State, Action}, Float64}()

        self.X2 = Dict{Tuple{State, Action}, Float64}()

        self.nloop_max = nloop_max
        self.nloop_min = nloop_min
        self.eps = eps

        self.runtime_max = runtime_max

        self.gamma_ = gamma_

        self.tree_policy = TreePolicy(tree_policy)

        if rollout == nothing
            self.rollout_type = :default
            self.rollout_func = rollout_default
        else
            self.rollout_type = rollout[1]
            self.rollout_func = rollout[2]
        end

        self.bReuse = bReuse

        self.visualizer = visualizer

        if self.visualizer != nothing
            if self.bReuse
                self.visualizer.b_hist_acc = true
            else
                self.visualizer.b_hist_acc = false
            end
        end

        return self
    end
end


function Generative(pm::MDP, s::State, a::Action)

    s_, r = MDP_.Generative(pm, s, a)

    return s_, r / pm.reward_norm_const
end


function default_policy(pm::MDP, s::State)
    
    a = pm.actions[rand(1:pm.nActions)]

    while !isFeasible(pm, s, a)
        a = pm.actions[rand(1:pm.nActions)]
    end

    return a
end


function rollout_default(alg::UCT, pm::MDP, s::State, d::Int64; debug::Int64 = 0)

    rgamma_ = 0.9

    if d == 0
        return 0
    end

    a = default_policy(pm, s)

    if debug > 2
        print(a, ", ")
    end

    s_, r = alg.Generative(pm, s, a)

    if isEnd(pm, s_)
        return r
    end

    return r + rgamma_ * rollout_default(alg, pm, s_, d - 1, debug = debug)
end


function simulate(alg::UCT, pm::MDP, s::State, d::Int64; debug::Int64 = 0)

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :start_sim, s)
    end

    if d == 0 || isEnd(pm, s)
        if !haskey(alg.T, s)
            for a in pm.actions
                alg.N[(s, a)] = 0

                if isFeasible(pm, s, a)
                    alg.Q[(s, a)] = 0
                else
                    alg.Q[(s, a)] = -Inf
                end

                alg.X2[(s, a)] = 0
            end

            alg.Ns[s] = 1
            alg.T[s] = true
        end

        if debug > 2
            if d != 0
                println("    hit end")
            end
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

        alg.Ns[s] = 1
        alg.T[s] = true

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
                    @assert var_[i] >= 0
                    RE[i] = sqrt(var_[i]) / abs(alg.Q[(s, a)])
                end

                if alg.tree_policy.bUCB1_tuned
                    Qv[i] = alg.Q[(s, a)] + sqrt(log(alg.Ns[s]) / alg.N[(s, a)] * min(1/4, var_[i] + sqrt(2 * log(alg.Ns[s]) / alg.N[(s, a)])))

                elseif alg.tree_policy.bUCB_V
                    Qv[i] = alg.Q[(s, a)] + sqrt(2 * var_[i] * log(alg.Ns[s]) / alg.N[(s, a)]) + alg.tree_policy.uv_c * 3 * log(alg.Ns[s]) / alg.N[(s, a)]

                end

            end
        end
    end

    a = pm.actions[argmax(Qv)]

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

    alg.N[(s, a)] += 1
    alg.Ns[s] += 1
    alg.Q[(s, a)] += (q - alg.Q[(s, a)]) / alg.N[(s, a)]
    alg.X2[(s, a)] += q * q

    if alg.visualizer != nothing
        updateTree(alg.visualizer, :after_sim, s, a, r * pm.reward_norm_const, q * pm.reward_norm_const, alg.N[(s, a)], alg.Ns[s], alg.Q[(s, a)] * pm.reward_norm_const)
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
        #println("N: ", alg.N)
        #println("Ns: ", alg.Ns)
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
    action = actions[rand(1:length(actions))]

    if alg.visualizer != nothing
        saveTree(alg.visualizer, pm)
    end

    return action, Q
end


function initialize(alg::UCT)

    alg.T = Dict{State, Bool}()
    alg.N = Dict{Tuple{State, Action}, Int64}()
    alg.Ns = Dict{State, Int64}()
    alg.Q = Dict{Tuple{State, Action}, Float64}()
    alg.X2 = Dict{Tuple{State, Action}, Float64}()
end


reinitialize(alg::UCT) = error("$(typeof(alg)) does not implement reinitialize()")


end


