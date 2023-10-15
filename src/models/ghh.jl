export GHH, GHHSolution,
    # compute_l̂, compute_ĥ, compute_x, compute_c, compute_u,
    # maximum_monotonic_and_convex, findmax_convex,
    # bellman_value, value_function_iterate
    solve


@kwdef struct GHH <: EconomicsModel
    # Preferences related parameters
    θ::Float64             = 1.0        # Relative risk aversion
    γ::Float64             = 2.0        # Labor disutility elasticity
    β::Float64             = 0.96       # Discount factor
    # Production-related parameters
    A::Float64             = 0.592      # TFP
    α::Float64             = 0.33       # Factor share of capital
                                        # Capacity constraints-related parameters
    B::Float64             = 0.075      # Depreciation rate
    ω::Float64             = 2          # Rate of capacity depreciation
                                        # Stochastic process
    # Capital-related parameters
    n::Integer             = 500        # Size of capital grid
    k_min::Float64         = 0.1        # Minimal value for capital grid
    k_max::Float64         = 6.0        # Maximum value of capital grid
    k_grid::Array{Float64} =            # Capital states
        range(k_min, k_max, length=n) |> collect
    # Investment-specific techonogy process
    m::Integer             = 3          # Size of technology shocks grid
    Θ::Float64             = .06        # Technology shocks magnitude
    ε_grid                 = [-Θ, 0, Θ] # Technology shocks grid
    ρ::Float64             = .7         # Auto correlation parameter
    Π::Matrix{Float64}     =            # Transition matrix between ε states
        [ρ         1-ρ     0
         (1-ρ)/2   ρ       (1-ρ)/2
         0         1-ρ     ρ      ]
end


struct GHHSolution <: ModelSolution
    value_function::Array{Float64}
    optimal_policy::Array{Int64}
    transition_matrix::Array{Int64}
end


function compute_l̂(k, ε, model)
    @unpack A, B, ω, α, θ = model
    return ( (1-α)^(ω-α) * α^α * A^ω * B^(-α) * k^(α*(ω-1)) * exp(ε*α) )^(1/(θ*(ω-α)+α*(ω-1)))
end


function compute_ĥ(k, ε, l̂, model)
    @unpack A, B, ω, α, θ = model
    return ( α * A/B * k^(α-1) * l̂^(1-α) * exp(ε) )^(1/ω-α)
end


function compute_c(k, k′, l̂, ε, model)
    @unpack θ, γ, ω, α = model
    return (1-α/ω)/(1-α)*l̂^(1+θ) + (k-k′)*exp(-ε)
end

# k_sol[t+1] - (1 - δ)*k_sol[t]
function compute_x(k, k′, ĥ, ε, model)
    @unpack B, ω, α = model
    return k′*exp(-ε) - (1 - B*(ĥ)^ω/ω)*k*exp(-ε)
end

function compute_u(k, k′, ε, model)
    @unpack θ, γ, ω, α = model
    l̂ = compute_l̂(k, ε, model)
    c = compute_c(k, k′, l̂, ε, model)
    CRRA_base = c - l̂^(1+θ)/(1+θ)
    return CRRA_base < 0 ? -Inf : CRRA_base^(1-γ)/(1-γ)
end


function policy_iterate(model::GHH, v_input::Array{Float64,2}, policy::Array{Int64,2}, u::Array{Float64,3};
                        max_howard_iter = 10, tol=1e-7)
    @unpack β, Π, k_grid, ε_grid, n, m = model
    # Initialize objects
    itr_h = 0
    diff = fill(Inf, 50, 3)
    # Optimal utility
    u_optimal = [u[i, j, policy[i, j]] for i in 1:n, j in 1:m]
    # Main loop
    while itr_h <= max_howard_iter && any(diff .>= tol)
        itr_h += 1
        v_old = v_input
        v_input = u_optimal + β * [v_input[policy[i,j],:]'*Π[:,j] for i in 1:n, j in 1:m]
        diff = abs.(v_old .- v_input)
    end
    return v_input
end


"""The Bellman equation"""
function bellman_value(model::GHH, v_guess::Array{Float64,2}, u; max_howard_iter=10)
    @unpack β, Π, k_grid, ε_grid, n, m = model
    # Expected value in the next period
    @inbounds  𝔼v = Π' * v_guess' |> v -> reshape(repeat(v, inner=(n,1)), n, m, n)
    # Compute the value function over all posible states
    v = u + β*𝔼v
    # Find the optimal desision for each of the current states
    v_iterated, policy = maximum_monotonic_and_convex(v, n, m)
    # Howards improvements: re-iterate the optimal policy
    v_policy_iterated = policy_iterate(model, v_iterated, policy, u,
                                       max_howard_iter=max_howard_iter)
    # Compute the difference
    diff = abs.(v_policy_iterated .- v_guess)
    return v_policy_iterated, diff, policy
end



"""Solve the GHH model with Value function iteration"""
function value_function_iterate(model::GHH; max_iter=1e3, tol=1e-7, max_howard_iter=10)
    @unpack β, Π, k_grid, ε_grid, n, m = model
    i = 1
    v_initial = zeros(n, m)
    # Compute the utility matrix
    u = [compute_u(k, k′, ε, model) for k in k_grid, ε in ε_grid, k′ in k_grid]
    # Starts the iteration process
    v_fn, diff, policy= bellman_value(model, v_initial, u, max_howard_iter=max_howard_iter)
    while any(diff .>= tol) && i <= max_iter
        v_fn, diff, policy = bellman_value(model, v_fn, u, max_howard_iter=max_howard_iter)
        i += 1
    end
    print("Value-function iteration terminated after ")
    printstyled(string(i) * " iterations.\n", color=:green)
    # Construct a transition matrix
    a_transition_matrix = zeros(n, n, m)
    for i_k in 1:n, i_ε in 1:m
        a_transition_matrix[i_k, policy[i_k, i_ε], i_ε] = 1
    end
    return GHHSolution(v_fn, policy, a_transition_matrix)
end


"""Solve the GHH model with VFI and Howard improvements, exploiting concavity and monotonicity"""
function solve(model::GHH; max_iter=1e3, tol=1e-7, max_howard_iter=10)
    value_function_iterate(model; max_iter=max_iter, tol=tol, max_howard_iter=max_howard_iter)
end


## Improvements fuctions
function findmax_convex(arr; shift=0)
    max_val = typemin(eltype(arr))
    idx = 0
    for i in 1:length(arr)
        if arr[i] > max_val
            idx += 1
            max_val = arr[i]
        else
            break
        end
    end
    return max_val, idx+shift
end


function maximum_monotonic_and_convex(arr::Array{Float64, 3}, n::Int64, m::Int64)
    value = Array{Float64}(undef, n, m)
    policy = Array{Int64}(undef, n, m)
    # i = 0
    for i_ε in 1:m
        i_k_low = 1
        for i_k in 1:n
            value[i_k, i_ε], i_k_low = findmax_convex(@view arr[i_k, i_ε, i_k_low:n]; shift=i_k_low-1)
            policy[i_k, i_ε] = i_k_low
        end
    end
    return value, policy
end
