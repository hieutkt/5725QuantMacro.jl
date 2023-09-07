export
    NeoClassicalGrowth,
    compute_steady_state,
    compute_lₜ, compute_cₜ, compute_λₜ, compute_yₜ, compute_rₜ,
    bracket_function, bisection_solve_for_kₜ₊₁, dekker_solve_for_kₜ₊₁,
    extended_path_solve,
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


"""Bisection algorithm to solve for kₜ₊₁ given values of kₜ and kₜ₊₂"""
function bisection_solve_for_kₜ₊₁(kₜ::Real, kₜ₊₂::Real, t::Integer, model::NeoClassicalGrowth;
                                  a=0.1, b=1.5,
                                  ε=1e-7, max_iter=1e2)
    f_a = bracket_function(kₜ, a, kₜ₊₂, t, model)
    f_b = bracket_function(kₜ, b, kₜ₊₂, t, model)
    # Make sure f(a) and f(b) have opposite signs
    @assert f_a*f_b < 0 "With f(a) = "*string(f_a)*"& f(b) = "*string(f_b)*": Not a proper bracket!!"
    # Compute the mid point
    m = (a + b) / 2
    f_m = bracket_function(kₜ, m, kₜ₊₂, t, model)
    # Initialize iterating
    iter = 0
    while abs(f_m) >= ε && iter <= max_iter
        iter += 1
        f_m*f_b >= 0 ? b = m : a = m
        m = (a + b) / 2
        f_m = bracket_function(kₜ, m, kₜ₊₂, t, model)
    end
    return m
end


"""Dekker's method to solve for kₜ₊₁ given values of kₜ and kₜ₊₂"""
function dekker_solve_for_kₜ₊₁(kₜ::Real, kₜ₊₂::Real, t::Integer, model::NeoClassicalGrowth;
                               a::Real=.1, b::Real=1.5,
                               ε=1e-7, max_iter=1e2)
    # Initiation
    f_a = bracket_function(kₜ, a, kₜ₊₂, t, model)
    f_b = bracket_function(kₜ, b, kₜ₊₂, t, model)
    # Make sure f(a) and f(b) have opposite signs
    @assert f_a*f_b < 0 "With f(a) = "*string(f_a)*"& f(b) = "*string(f_b)*": Not a proper bracket!!"
    # Set the last iteration of b to a
    c, f_c = a, f_a
    # Initialize iterating
    iter = 0
    while abs(f_b) >= ε && iter <= max_iter
        iter += 1
        # Make sure b is the 'better' guess than a
        abs(f_b) > abs(f_a) ? (a, f_a, b, f_b) = (b, f_b, a, f_a) : nothing
        # Compute the mid point
        m = (a + b) / 2
        # Compute the secant point; fall back to m in case s undefined
        f_b == f_c ? s = m : s = b - f_b * (b-c)/(f_b-f_c)
        # Update values:
        # b becomes c, the last iterations' best guess
        c, f_c = b, f_b
        # If s is between m and b => s becomes new b, else take m
        m < s < b || b < s < m ? b = s : b = m
        f_b = bracket_function(kₜ, b, kₜ₊₂, t, model)
        # If the b changes sign, assign the previous iteration as a
        f_b*f_c < 0 ? (a, f_a) = (c, f_c) : nothing
    end
    return b
end


"""Solve the neo-classical growth model using the extended path method"""
function extended_path_solve(k_guess, model;
                             solve_method = dekker_solve_for_kₜ₊₁,
                             ε=1e-7, max_iter=1e5)
    @unpack T = model
    # Declare the iterate methods
    iterate_method(kₜ, kₜ₊₂, t) = solve_method(kₜ, kₜ₊₂, t, model)
    # Initialize the loop
    iter = 0
    continue_condition = true
    k_iterations = [k_guess]
    # Iterate as long as the pointwise distance between any time period is > ε
    while continue_condition && iter < max_iter
        # Let's keep updating the situation every 100 iterations
        iter += 1
        iter % 100 == 0 ? print("Iteration ", string(iter), "s \r") : nothing
        k_new = map(iterate_method, k_guess[1:T-2], k_guess[3:T], 2:T-1)
        # Whether to continue
        continue_condition = any(abs.(k_new .- k_guess[2:T-1]) .>= ε)
        # Update the guess
        k_guess = cat(k_guess[1], k_new, k_guess[T], dims=1)
        push!(k_iterations, k_guess)
    end
    # Return the results
    println("Extended path method terminated after ", string(iter), " iterations with pointwise precision ε=", ε)
    return k_iterations
end


"""Main interface to solving models"""
function solve(model::NeoClassicalGrowth;
               solve_method = dekker_solve_for_kₜ₊₁,)
    kₛₛ, _ = compute_steady_state(model)
    k_guess = LinRange(model.k₀, kₛₛ, model.T) |> collect
    return extended_path_solve(k_guess, model; solve_method=solve_method)
end
