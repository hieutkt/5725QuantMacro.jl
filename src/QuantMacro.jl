module QuantMacro

using LinearAlgebra, SparseArrays
using Distributions

using UnPack: @unpack


# Write your package code here.

abstract type EconomicsModel end
abstract type ModelSolution end


include("utility.jl")
include("models/neoclassicalgrowth.jl")
include("models/ghh.jl")
include("utils.jl")

end
