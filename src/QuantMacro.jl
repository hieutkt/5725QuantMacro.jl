module QuantMacro

using LinearAlgebra, SparseArrays

using UnPack: @unpack


# Write your package code here.

abstract type EconomicsModel end

include("models/neoclassicalgrowth.jl")
include("utils.jl")

end
