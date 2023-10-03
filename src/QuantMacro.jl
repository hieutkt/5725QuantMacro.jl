module QuantMacro

using LinearAlgebra, SparseArrays
using Distributions

using UnPack: @unpack


# Write your package code here.

abstract type EconomicsModel end
abstract type ModelSolution end


include("utility.jl")
include("utils.jl")
include("non-linear-solvers.jl")
# HW1
include("models/neoclassicalgrowth.jl")
# HW2
include("models/ghh.jl")
# HW4
include("models/incomefluctuations.jl")

end
