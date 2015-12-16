# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 10/15/2015

VERSION >= v"0.4" && __precompile__(false)


module MineField_

import Base: isequal, ==, hash, copy, string
import Base: start, done, next

export MineField, MFState, MFAction
export MFStateIter
export nextState, reward, Generative, isEnd, isFeasible
export tranProb
export computeExpectedRewardMap


using MDP_
using Dist_

using Distributions
using Base.Test


import MDP_: nextState, reward, Generative, isEnd, isFeasible, tranProb
import MDP_: MDP, State, Action


immutable MFState <: State

    x::Int64
    y::Int64
end

function string(s::MFState)

    return "($(s.x), $(s.y))"
end

type MFStateIter

    nx::Int64
    ny::Int64

    function MFStateIter(nx, ny)

        self = new()

        self.nx = nx
        self.ny = ny

        return self
    end
end

function start(iter::MFStateIter)

    state = (1, 1)
end

function done(iter::MFStateIter, state)

    x, y = state

    if y > iter.ny
        return true
    end

    return false
end

function next(iter::MFStateIter, state)

    x, y = state

    item = MFState(x, y)

    if x == iter.nx
        x = 1
        y += 1
    else
        x += 1
    end

    return item, (x, y)
end


immutable MFAction <: Action

    action::Symbol
end

function string(a::MFAction)

    return string(a.action)
end


type MineField <: MDP

    seed::Union{Int64, Void}

    nx::Int64
    ny::Int64

    states::MFStateIter
    nStates::Int64

    actions::Vector{MFAction}
    nActions::Int64

    rover_init_loc::Tuple{Int64, Int64}

    destination::Tuple{Int64, Int64}

    Reward::Any

    reward_min::Float64
    reward_max::Float64

    reward_norm_const::Float64


    function MineField(nx::Int64, ny::Int64; seed::Union{Int64, Void} = nothing, Reward::Any = nothing)

        self = new()

        if seed == nothing
            self.seed = round(Int64, time())
        else
            self.seed = seed
        end

        # Note: can't create rng since Distributions.jl does not support rng argument in its rand()
        srand(self.seed)

        self.nx = nx
        self.ny = ny

        self.states = MFStateIter(nx, ny)
        self.nStates = nx * ny

        self.actions = [MFAction(:up), MFAction(:right)]
        self.nActions = length(self.actions)

        self.rover_init_loc = (1, 1)

        self.destination = (nx, ny)

        if Reward != nothing
            self.Reward = Reward
        else
            self.Reward = Array(FixedValue, nx, ny)
            D = Truncated(Normal(-10, 4), -Inf, -1)
            for i = 1:nx
                for j = 1:ny
                    self.Reward[i, j] = FixedValue(round(Int64, rand(D)))
                end
            end
            self.Reward[self.rover_init_loc...] = FixedValue(0)
            self.Reward[self.destination...] = FixedValue(0)
        end

        self.reward_min = minimum([minimum(self.Reward[i, j]) for i = 1:nx, j = 1:ny])
        self.reward_max = maximum([maximum(self.Reward[i, j]) for i = 1:nx, j = 1:ny])

        self.reward_norm_const = 1.

        return self
    end
end


function computeExpectedRewardMap(Reward)

    nx, ny = size(Reward)

    R = zeros(Float64, nx, ny)

    for i = 1:nx
        for j = 1:ny
            R[i, j] = mean(Reward[i, j])
        end
    end

    return R
end


function isFeasible(mf::MineField, s::MFState, a::MFAction)

    if a.action == :up
        if s.y == mf.ny
            return false
        end
    elseif a.action == :down
        if s.y == 1
            return false
        end
    elseif a.action == :left
        if s.x == 1
            return false
        end
    elseif a.action == :right
        if s.x == mf.nx
            return false
        end
    end

    return true
end


# P(s' | s, a)
function tranProb(mf::MineField, s::MFState, a::MFAction, s_::MFState)

    @test isFeasible(mf, s, a)

    if a.action == :up
        if s.x == s_.x && s.y + 1 == s_.y
            return 1.
        end
    elseif a.action == :down
        if s.x == s_.x && s.y - 1 == s_.y
            return 1.
        end
    elseif a.action == :left
        if s.x - 1 == s_.x && s.y == s_.y
            return 1.
        end
    elseif a.action == :right
        if s.x + 1 == s_.x && s.y == s_.y
            return 1.
        end
    end

    return 0.
end


# s' ~ P(S | s, a)
function nextState(mf::MineField, s::MFState, a::MFAction)

    @test isFeasible(mf, s, a)

    x = s.x
    y = s.y

    if a.action == :up
        x_ = x
        y_ = y + 1
    elseif a.action == :down
        x_ = x
        y_ = y - 1
    elseif a.action == :left
        x_ = x - 1
        y_ = y
    elseif a.action == :right
        x_ = x + 1
        y_ = y
    end

    s_ = MFState(x_, y_)

    return s_
end


function isEnd(mf::MineField, s::MFState)

    if (s.x, s.y) == mf.destination
        return true
    else
        return false
    end
end


# R(s, a)
function reward(mf::MineField, s::MFState, a::MFAction, s_::MFState)

    r = rand(mf.Reward[s_.x, s_.y])

    return r
end


# (s', r) ~ G(s, a)
function Generative(mf::MineField, s::MFState, a::MFAction)

    s_ = nextState(mf, s, a)
    r = reward(mf, s, a, s_)

    return s_, r
end


function isequal(s1::MFState, s2::MFState)

    return isequal((s1.x, s1.y), (s2.x, s2.y))
end

function ==(s1::MFState, s2::MFState)

    return (s1.x, s1.y) == (s2.x, s2.y)
end

function hash(s::MFState, h::UInt64 = zero(UInt64))

    return hash((s.x, s.y), h)
end

function copy(s::MFState)

    return MFState(s.x, s.y)
end


function isequal(a1::MFAction, a2::MFAction)

    return isequal(a1.action, a2.action)
end

function ==(a1::MFAction, a2::MFAction)

    return (a1.action == a2.action)
end

function hash(a::MFAction, h::UInt64 = zero(UInt64))

    return hash(a.action, h)
end


end


