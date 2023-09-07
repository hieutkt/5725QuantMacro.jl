module QuantMacro

using LinearAlgebra, SparseArrays

using UnPack: @unpack
using CairoMakie, ColorSchemes


# Write your package code here.

abstract type EconomicsModel end

include("models/neoclassicalgrowth.jl")
include("utils.jl")

end
