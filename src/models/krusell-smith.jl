export KrusellSmith,
    # compute_r, compute_w, Euler_residual, compute_K₂,
    solve

@kwdef struct KrusellSmith <: EconomicsModel
    # PREFERENCES
    σ::Float64 = 4.0            # CRRA Parameter
    u = CRRAUtility(σ)
    u′ = marginal(u)
    β::Float64 = 0.9            # Discount rate
    # PROUCTION
    α::Float64 = 0.36           # Factor price for capital
    z::Float64 = 1.0            # Productivity
    # CAPITAL
    a₁_min::Float64 =  0.0      # Minimal asset
    a₂_min::Float64 = -2.0
    a_max::Float64  =  2.0      # Maximum asset
    K₁::Float64 = 10.0
    # ENDOWNMENT
    s₁_grid::Array{Float64} = [0.75, 1.25]
    s₂_grid::Array{Float64} = [0.9 , 1.1 ]
end


"""Compute interest rate given aggregate capital K"""
function compute_r(mdl::KrusellSmith, K)
    @unpack α, z = mdl
    return α*z*K^(α-1)
end


"""Compute wage level given aggregate capital K"""
function compute_w(mdl::KrusellSmith, K)
    @unpack α, z = mdl
    return (1-α)*z*K^α
end


"""Compute the Euler residual"""
function Euler_residual(mdl, a₁, s₁, a₂, K₂)
    @unpack u′, β, K₁, s₂_grid = mdl
    # Compute the prices
    r₁ = compute_r(mdl, K₁)
    w₁ = compute_w(mdl, K₁)
    R₂ = compute_r(mdl, K₂)
    w₂ = compute_w(mdl, K₂)
    # Compute The Euler residual
    lhs = u′(r₁ * a₁ + w₁ * s₁ - a₂)
    rhs = β*R₂*mean([mdl.u′(R₂ * a₂ + w₂ * s₂) for s₂ in s₂_grid])
    return lhs - rhs
end


"""Compute the the aggregate level of capital in the second period"""
function compute_K₂(mdl, K₂_guess; a1_gridsize = 100)
    @unpack a₁_min, a₂_min, a_max = mdl
    a₁_grid = LinRange(a₁_min, a_max, a1_gridsize) |> collect
    a₂_optimal = [brent_solve_f(a₂ ->  Euler_residual(mdl, a₁, s₁, a₂, K₂_guess), a₂_min, a_max)
                  for a₁ ∈ a₁_grid, s₁ ∈ mdl.s₁_grid]
    return mean(a₂_optimal)
end


"""Solve for aggregate level of capital that is consistent"""
function solve(mdl::KrusellSmith; a = 0.001, b = 1)
    return brent_solve_f(K₂ -> (compute_K₂(mdl, K₂) - K₂), a, b, verbose = true)
end
