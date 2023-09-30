export
    NeoClassicalGrowth,
    compute_steady_state,
    # compute_lₜ, compute_cₜ, compute_λₜ, compute_yₜ, compute_rₜ,
    # bracket_function,
    # extended_path_solve,
    solve


Base.@kwdef struct NeoClassicalGrowth <: EconomicsModel
    σ::Float64          = 2.                 # Elasticity of substitution
    β::Float64          = 0.96               # Discount factor
    θ::Float64          = 0.33               # Factor share of capital
    δ::Float64          = 0.081              # Depreciation rate
    η::Float64          = 1.8                # Preferences for labor
    T::Integer          = 100                # The number of time periods
    A::Array{Float64,1} = repeat([0.624], T) # TFP
    k₀::Float64         = 0.65               # Initial capital
end


"""Compute the steady state given a NeoClassicalGrowth model"""
function compute_steady_state(model::NeoClassicalGrowth)
    @unpack σ, β, θ, δ, η, A, T = model
    kₛₛ_over_lₛₛ = ((1/β - (1 - δ))/(A[T]*θ))^(1/(θ - 1))
    lₛₛ = (A[T] * (1 - θ) * kₛₛ_over_lₛₛ^(θ) )^(1/(η - 1))
    kₛₛ = kₛₛ_over_lₛₛ * lₛₛ
    return kₛₛ, lₛₛ
end


"""Compute the output given capital and labor inputs"""
function compute_yₜ(kₜ, lₜ, t, model)
    @unpack A, θ = model
    return A[t] *  kₜ^θ * lₜ^(1-θ)
end


"""Compute labor given a level of capital"""
function compute_lₜ(kₜ, t, model)
    @unpack θ, η, A = model
    return (A[t] * (1 - θ) * kₜ^θ)^(1/(θ + η - 1))
end


"""Compute consumption from the budget contraint"""
function compute_cₜ(kₜ, kₜ₊₁, lₜ, t, model)
    @unpack θ, δ, A  = model
    return A[t] * kₜ^θ * lₜ^(1-θ) + (1 - δ)*kₜ - kₜ₊₁
end


"""Compute the marginal utility over consumption"""
function compute_λₜ(cₜ, lₜ, model)
    @unpack η, σ = model
    return (cₜ - (lₜ^η)/η)^(-σ)
end


"""Compute interest rates as the marginal capital productivity"""
function compute_rₜ(kₜ, lₜ, t, model)
    @unpack θ, A, δ = model
    return A[t] * θ * (kₜ/lₜ)^(θ-1) - δ
end


"""Interate function to be used in the bisection algorithm"""
function bracket_function(kₜ, kₜ₊₁, kₜ₊₂, t, model)
    @unpack β, θ, δ, A, T = model
    lₜ   = compute_lₜ(kₜ  ,             t  , model)
    lₜ₊₁ = compute_lₜ(kₜ₊₁,             t+1, model)
    cₜ   = compute_cₜ(kₜ  , kₜ₊₁, lₜ  , t  , model)
    cₜ₊₁ = compute_cₜ(kₜ₊₁, kₜ₊₂, lₜ₊₁, t+1, model)
    λₜ   = compute_λₜ(cₜ  ,       lₜ  ,      model)
    λₜ₊₁ = compute_λₜ(cₜ₊₁,       lₜ₊₁,      model)
    return A[t+1]*θ*(kₜ₊₁/lₜ₊₁)^(θ-1) + (1 - δ) - λₜ/(β*λₜ₊₁)
end


"""Solve the neo-classical growth model using the extended path method"""
function extended_path_solve(k_guess, model;
                             solve_method = brent_solve_f,
                              a::Float64=.1, b::Float64=1.5,
                             ε=1e-7, max_iter=1e5, verbose=false)
    @unpack T = model
    # Declare the iterate methods
    iterate_method(kₜ, kₜ₊₂, t) = solve_method(x -> bracket_function(kₜ, x, kₜ₊₂, t, model), a, b)
    # Initialize the loop
    iter = 0
    continue_condition = true
    k_iterations = [k_guess]
    # Iterate as long as the pointwise distance between any time period is > ε
    while continue_condition && iter < max_iter
        # Let's keep updating the situation every 100 iterations
        iter += 1
        if verbose && iter % 100 == 0
             print("Iteration ", string(iter), "s \r")
        end
        k_new = map(iterate_method, k_guess[1:T-2], k_guess[3:T], 2:T-1)
        # Whether to continue
        continue_condition = any(abs.(k_new .- k_guess[2:T-1]) .>= ε)
        # Update the guess
        k_guess = cat(k_guess[1], k_new, k_guess[T], dims=1)
        push!(k_iterations, k_guess)
    end
    # Return the results
    return k_iterations
end


"""Main interface to solving models"""
function solve(model::NeoClassicalGrowth;
                solve_method = brent_solve_f,
                a::Float64=.1, b::Float64=1.5)
    kₛₛ, _ = compute_steady_state(model)
    k_guess = LinRange(model.k₀, kₛₛ, model.T) |> collect
    return extended_path_solve(k_guess, model; solve_method=solve_method, a=a, b=b)
end
