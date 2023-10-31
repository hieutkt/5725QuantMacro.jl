module QuantMacro

using LinearAlgebra, SparseArrays
using Distributions

using UnPack: @unpack

# For clean display of model objects
using Term: termshow

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
# HW5
include("models/sovereign-default.jl")
# HW6
include("models/household-lifecycle.jl")
# HW7
include("models/household-finance.jl")


# Misc
include("misc/display.jl")

end
