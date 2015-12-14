# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 10/15/2015

module MineFieldVisualizer_

export MineFieldVisualizer, visInit, visUpdate, updateAnimation, saveAnimation


using Visualizer_
using MineField_

using Util

using PyCall
using PyPlot

@pyimport matplotlib.animation as ani

import Visualizer_: visInit, visUpdate, updateAnimation, saveAnimation


type MineFieldVisualizer <: Visualizer

    fig::Union{Figure, Void}
    ax::Union{PyObject, Void}

    cell_size::Float64

    artists::Union{Vector{PyObject}, Void}

    ims::Vector{Any}

    wait::Bool


    function MineFieldVisualizer(;wait = false)

        self = new()

        ion()

        self.fig = nothing
        self.ax = nothing

        self.artists = nothing

        self.ims = Any[]

        self.wait = wait
        
        return self
    end
end


function visInit(mfv::MineFieldVisualizer, mf::MineField, expected_reward_map::Union{Array{Float64, 2}, Void} = nothing, path::Union{Vector{Tuple{Int64, Int64}}, Void} = nothing)

    if mfv.fig == nothing
        fig = figure(facecolor = "white")

        ax = fig[:add_subplot](111)
        ax[:set_aspect]("equal")
        ax[:set_xlim](0.5, mf.nx + 0.5)
        ax[:set_ylim](0.5, mf.ny + 0.5)
        ax[:set_xticks](collect(0.5:(mf.nx + 0.5)))
        ax[:set_yticks](collect(0.5:(mf.ny + 0.5)))
        ax[:set_xticklabels]([])
        ax[:set_yticklabels]([])
        ax[:grid](true)
        ax[:set_title]("MineField")

        mfv.fig = fig
        mfv.ax = ax
    else
        fig = mfv.fig
        ax = mfv.ax

        for artist in mfv.artists
            artist[:set_visible](false)
        end
    end

    bbox = ax[:get_window_extent]()[:transformed](fig[:dpi_scale_trans][:inverted]())   # inches
    ax_width = bbox[:width] * 72    # points
    ax_height = bbox[:height] * 72  # points
    mfv.cell_size = ax_width / mf.nx    # points

    artists = PyObject[]

    rover_init_marker = ax[:plot](mf.rover_init_loc[1], mf.rover_init_loc[2], "sk", markersize = mfv.cell_size / 5)
    append!(artists, rover_init_marker)

    dest_marker_text = ax[:text](mf.destination[1], mf.destination[2], "X", size = mfv.cell_size / 4, horizontalalignment = "center", verticalalignment = "center")
    push!(artists, dest_marker_text)

    if expected_reward_map != nothing
        for i = 1:mf.nx
            for j = 1:mf.ny
                if !((i == 1 && j == 1) || (i == mf.nx && j == mf.ny))
                    reward_text = ax[:text](i, j, neat(expected_reward_map[i, j]), size = mfv.cell_size / 8, horizontalalignment = "center", verticalalignment = "center")
                    push!(artists, reward_text)
                end
            end
        end
    end

    if path != nothing
        rover_path = ax[:plot](map(x -> x[1], path), map(x -> x[2], path), "r")
        append!(artists, rover_path)
    end

    fig[:canvas][:draw]()

    mfv.artists = artists

    return mfv
end


function visUpdate(mfv::MineFieldVisualizer, mf::MineField, ts::Union{Int64, Void} = nothing, s::Union{MFState, Void} = nothing, a::Union{MFAction, Void} = nothing, r::Union{Int64, Float64} = 0, R::Union{Int64, Float64} = 0, ER::Union{Int64, Float64} = 0)

    fig = mfv.fig
    ax = mfv.ax

    if ts == nothing
        text = ax[:text](0.5, -0.02, "$(mf.nx) x $(mf.ny)", horizontalalignment = "center", verticalalignment = "top", transform = ax[:transAxes])
        push!(mfv.artists, text)
    else
        if ts == 0
            text = ax[:text](0.5, -0.02, "timestep: 0, reward: 0, total reward: 0", horizontalalignment = "center", verticalalignment = "top", transform = ax[:transAxes])
        else
            action = string(a)
            text = ax[:text](0.5, -0.02, "timestep: $ts, action: $action, reward: $r, total reward: $R, ER: $ER", horizontalalignment = "center", verticalalignment = "top", transform = ax[:transAxes])
        end
        push!(mfv.artists, text)

        rover_marker = ax[:plot](s.x, s.y, "sm", markersize = mfv.cell_size / 4)
        append!(mfv.artists, rover_marker)
    end

    fig[:canvas][:draw]()

    return mfv
end


function updateAnimation(mfv::MineFieldVisualizer, ts::Union{Int64, Void} = nothing; bSaveFrame::Bool = false, filename::ASCIIString = "minefield.png")

    append!(mfv.ims, Any[mfv.artists])

    if bSaveFrame
        if ts == nothing
            savefig(filename, transparent = false)
        else
            base, ext = splitext(filename)
            savefig(base * "_" * string(ts) * "." * ext, transparent = false)
        end
    end

    if mfv.wait
        readline()
    end
end


function saveAnimation(mfv::MineFieldVisualizer; interval::Int64 = 1000, repeat::Bool = false, filename::ASCIIString = "minefield.mp4")

    if repeat || mfv.wait
        readline()
        println("save animation")
    end

    im_ani = ani.ArtistAnimation(mfv.fig, mfv.ims, interval = interval, repeat = repeat, repeat_delay = interval * 5)
    im_ani[:save](filename)

    if repeat || mfv.wait
        readline()
    end
end


end


