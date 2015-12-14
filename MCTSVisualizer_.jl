# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 12/12/2015

module MCTSVisualizer_

export MCTSVisualizer, initTree, updateTree, saveTree


using MDP_

using JSON


type MCTSVisualizer

    sim_file::ASCIIString

    T_sim::Dict{ASCIIString, Any}
    T_sim_curr::Dict{ASCIIString, Any}
    T_sim_stack::Vector{Dict{ASCIIString, Any}}


    function MCTSVisualizer(; sim_file::ASCIIString = "sim.json")

        self = new()

        self.sim_file = sim_file

        return self
    end
end


function initTree(vis::MCTSVisualizer)

    vis.T_sim = Dict{ASCIIString, Any}()
    vis.T_sim_curr = vis.T_sim
    vis.T_sim_stack = Dict{ASCIIString, Any}[]
end


function updateTree(vis::MCTSVisualizer, where::Symbol, args...)

    if where == :start_sim
        s, = args
        st = string(s)

        T_sim_curr = vis.T_sim_curr

        if !haskey(T_sim_curr, st)
            T_sim_curr[st] = Dict{ASCIIString, Any}()
            T_sim_curr[st]["actions"] = Dict{ASCIIString, Any}()
            T_sim_curr[st]["n"] = 0
            T_sim_curr[st]["N"] = 0
        end

    elseif where == :before_sim
        s, a = args

        st = string(s)
        act = string(a)

        T_sim_curr = vis.T_sim_curr

        if !haskey(T_sim_curr[st]["actions"], act)
            T_sim_curr[st]["actions"][act] = Dict{ASCIIString, Any}()
            T_sim_curr[st]["actions"][act]["states"] = Dict{ASCIIString, Any}()
            T_sim_curr[st]["actions"][act]["n"] = 0
            T_sim_curr[st]["actions"][act]["r"] = 0.
            T_sim_curr[st]["actions"][act]["N"] = 0
            T_sim_curr[st]["actions"][act]["Q"] = 0.
        end

        push!(vis.T_sim_stack, T_sim_curr)
        vis.T_sim_curr = T_sim_curr[st]["actions"][act]["states"]

    elseif where == :after_sim
        s, a, r, q, Ns, N, Q = args

        st = string(s)
        act = string(a)

        vis.T_sim_curr = pop!(vis.T_sim_stack)
        T_sim_curr = vis.T_sim_curr

        T_sim_curr[st]["n"] += 1
        T_sim_curr[st]["actions"][act]["n"] += 1
        T_sim_curr[st]["actions"][act]["r"] += (r - T_sim_curr[st]["actions"][act]["r"]) / T_sim_curr[st]["actions"][act]["n"]

        T_sim_curr[st]["N"] = Ns
        T_sim_curr[st]["actions"][act]["N"] = N
        T_sim_curr[st]["actions"][act]["Q"] = Q
    end
end


function saveTree(vis::MCTSVisualizer, pm::MDP)

    function process(Tin, Tout, level)

        if rem(level, 2) == 0
            for (state, node) in Tin
                node_ = Dict{ASCIIString, Any}()
                node_["state"] = state
                node_["n"] = node["n"]
                node_["N"] = node["N"]
                node_["actions"] = Any[]

                push!(Tout, node_)

                process(node["actions"], node_["actions"], level + 1)
            end

        elseif rem(level, 2) == 1
            for a in pm.actions
                action = string(a)
                if haskey(Tin, action)
                    node = Tin[action]
                else
                    continue
                end

                node_ = Dict{ASCIIString, Any}()
                node_["action"] = action
                node_["n"] = node["n"]
                node_["r"] = node["r"]
                node_["N"] = node["N"]
                node_["Q"] = node["Q"]
                node_["states"] = Any[]

                push!(Tout, node_)

                process(node["states"], node_["states"], level + 1)
            end

        end
    end

    Tout = Any[]

    process(vis.T_sim, Tout, 0)

    f = open(vis.sim_file, "w")
    JSON.print(f, Tout[1], 2)
    close(f)

    return Tout
end


end


