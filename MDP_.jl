# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 10/15/2015

VERSION >= v"0.4" && __precompile__(false)


# Markov Decision Process
module MDP_

export MDP, State, Action
export nextState, reward, Generative, isEnd, isFeasible
export tranProb

abstract MDP
abstract State
abstract Action

nextState(pm::MDP) = error("$(typeof(pm)) does not implement nextState()")
reward(pm::MDP) = error("$(typeof(pm)) does not implement reward()")
Generative(pm::MDP) = error("$(typeof(pm)) does not implement Generative()")
isEnd(pm::MDP) = error("$(typeof(pm)) does not implement isEnd()")
isFeasible(pm::MDP) = error("$(typeof(pm)) does not implement isFeasible()")
tranProb(pm::MDP) = error("$(typeof(pm)) does not implement tranProb()")

end


