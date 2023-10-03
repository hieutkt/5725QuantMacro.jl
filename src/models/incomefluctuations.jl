export IncomeFluctuation, RedistributiveTaxation,
    endogeneous_grid_iteration

Base.@kwdef struct IncomeFluctuation <: EconomicsModel
    # Preferences related parameters
    σ::Float64             = 4.0        # Relative risk aversion
    β::Float64             = 0.93       # Discount rate
    # Production-related parameters
    R::Float64             = 1.01       # Gross interest rate
    # Asset-related parameters
    M::Integer             = 500        # Size of asset grid
    a_min::Float64         = 0.0        # Minimal value for asset grid
    a_max::Float64         = 6.0        # Maximum value of asset grid
    a_grid::Array{Float64} =            # asset states
        LinRange(a_min, a_max, M)
    # Income: AR(1) parameters
    ρ::Float64             = .98
    σ²_ε::Float64          = 0.04
    # Income grid and transition matrix
    N::Integer             = 31          # Size of income grid
    σ²_y::Float64          = 0.49        # In SE, σ²_ε / (1 - ρ^2)
    # Automatically create income grids and the transition matrix
    log_y_grid::Array{Float64} = LinRange(-3*sqrt(σ²_y), 3*sqrt(σ²_y), N)
    y_grid::Array{Float64} = exp.(log_y_grid)
    Π::Matrix{Float64}     =
        tauchen_discretize(log_y_grid, ρ, sqrt(σ²_ε))
end


Base.@kwdef struct RedistributiveTaxation <: EconomicsModel
    # Preferences related parameters
    σ::Float64             = 4.0        # Relative risk aversion
    β::Float64             = 0.93       # Discount rate
    # Production-related parameters
    R::Float64             = 1.01       # Gross interest rate
    # Asset-related parameters
    M::Integer             = 500        # Size of asset grid
    a_min::Float64         = 0.0        # Minimal value for asset grid
    a_max::Float64         = 6.0        # Maximum value of asset grid
    a_grid::Array{Float64} =            # asset states
        LinRange(a_min, a_max, M)
    # Income: AR(1) parameters
    ρ::Float64             = .98
    σ²_ε::Float64          = 0.04
    # Income grid and transition matrix
    N::Integer             = 31          # Size of income grid
    σ²_y::Float64          = 0.49        # In SE, σ²_ε / (1 - ρ^2)
    # Automatically create income grids and the transition matrix
    log_y_grid::Array{Float64} = LinRange(-3*sqrt(σ²_y), 3*sqrt(σ²_y), N)
    y_grid::Array{Float64} = exp.(log_y_grid)
    Π::Matrix{Float64}     =
        tauchen_discretize(log_y_grid, ρ, sqrt(σ²_ε))
    income_dist            = ^(Π, 10000)[1, :] # Invariant distribution
    # Tax
    τ                      = 0.3         # Tax rates
    ŷ_grid                 =             # Disposable income
        y_grid.^(1-τ)  * (income_dist ⋅ y_grid)^τ
end


## Solving
mu(c, σ) = c > 0 ? c^(-σ) : Inf
inverse_mu(u, σ) = u > 0 ? u^(-1/σ) : Inf


function compute_a(mdl::IncomeFluctuation, i, j, policy_guess, a_grid)
    @unpack y_grid, R, β, Π, σ, N,  a_max = mdl
    EE_rhs = R*β* Π[j, :] ⋅ mu.(R * a_grid[i] .+ y_grid .-  policy_guess[:,i], σ)
    a = (inverse_mu(EE_rhs, σ) - y_grid[j] + a_grid[i]) / R
    return a
end


function compute_a(mdl::RedistributiveTaxation, i, j, policy_guess, a_grid)
    @unpack ŷ_grid, R, β, Π, σ, N, a_max, income_dist, τ = mdl
    EE_rhs = R*β* Π[j, :] ⋅ mu.(R * a_grid[i] .+ ŷ_grid .-  policy_guess[:,i], σ)
    a = (inverse_mu(EE_rhs, σ) - ŷ_grid[j] + a_grid[i]) / R
    return a
end


function update_policy_edogeneous(mdl::IncomeFluctuation, policy_guess::Array{Float64,2}, a_grid::Array{Float64,1};
                                  interpolation_method=interpolate_linear)
    @unpack y_grid, R, β, Π, σ, N, M, a_min, a_max = mdl
    @assert map(issorted, eachslice(policy_guess, dims = 1)) |> all "Uh-oh! Policy function not monotonic!"
    # Compute the current a that would facilitate the policy guess
    inverse_choice = [compute_a(mdl, i, j, policy_guess, a_grid) for j in 1:N, i in 1:M]
    # Look at the choice of the richest guy
    if all(.<(0), policy_guess[:,M] .- inverse_choice[:,M])
        # If all of them are disaving, keep the grid size fixed
        new_a_max = max(policy_guess...)
    else
        # If some of them are still accumulating, extend the grid
        new_a_max = maximum(filter(!=(Inf), inverse_choice))
    end
    # New grid
    new_a_grid = LinRange(a_min, new_a_max, M) |> collect
    # Interpolate the guess
    new_policy_guess = [interpolation_method(new_a_grid[i], inverse_choice[j,:], a_grid) for j in 1:N, i in 1:M]
    # Make sure that the new grid is feasible
    new_policy_guess[findall(<(a_min), new_policy_guess)] .= a_min
    a_max_feasible = [R * a_grid[i] + y_grid[j] for j in 1:N, i in 1:M]
    new_policy_guess[new_policy_guess .> a_max_feasible] .= a_max_feasible[new_policy_guess .> a_max_feasible] .- 1e-7
    return new_policy_guess, new_a_grid
end


function update_policy_edogeneous(mdl::RedistributiveTaxation, policy_guess::Array{Float64,2}, a_grid::Array{Float64,1};
                                  interpolation_method=interpolate_linear)
    @unpack ŷ_grid, R, β, Π, σ, N, M, a_min, a_max = mdl
    @assert map(issorted, eachslice(policy_guess, dims = 1)) |> all "Uh-oh! Policy function not monotonic!"
    # Compute the current a that would facilitate the policy guess
    inverse_choice = [compute_a(mdl, i, j, policy_guess, a_grid) for j in 1:N, i in 1:M]
    # Look at the choice of the richest guy
    if all(.<(0), policy_guess[:,M] .- inverse_choice[:,M])
        # If all of them are disaving, keep the grid size fixed
        new_a_max = max(policy_guess...)
    else
        # If some of them are still accumulating, extend the grid
        new_a_max = maximum(filter(!=(Inf), inverse_choice))
    end
    # New grid
    new_a_grid = LinRange(a_min, new_a_max, M) |> collect
    # Interpolate the guess
    new_policy_guess = [interpolation_method(new_a_grid[i], inverse_choice[j,:], a_grid) for j in 1:N, i in 1:M]
    # Make sure that the new grid is feasible
    new_policy_guess[findall(<(a_min), new_policy_guess)] .= a_min
    a_max_feasible = [R * a_grid[i] + ŷ_grid[j] for j in 1:N, i in 1:M]
    new_policy_guess[new_policy_guess .> a_max_feasible] .= a_max_feasible[new_policy_guess .> a_max_feasible] .- 1e-7
    return new_policy_guess, new_a_grid
end


function endogeneous_grid_iteration(mdl::IncomeFluctuation; max_iter=10e5, tol=1e-6,
                                    interpolation_method=interpolate_linear)
    @unpack a_grid, y_grid, R, β, Π, σ, N, M = mdl
    # An intial guess is just linear
    policy_guess = reshape(repeat(a_grid, inner=(N,1)), N, M)
    # Update policy
    iter = 0
    diff = Inf
    while any(diff .> tol) && iter <= max_iter
        policy_previous, new_a_grid = update_policy_edogeneous(mdl, policy_guess, a_grid, interpolation_method=interpolation_method)
        diff = policy_previous .- policy_guess .|> abs
        # diff = a_grid .- new_a_grid .|> abs
        iter += 1
        print("Iteration: $(iter), diff = $(sum(diff))              \r")
        policy_guess, a_grid = policy_previous, new_a_grid
    end
    return policy_guess, a_grid
end


function endogeneous_grid_iteration(mdl::RedistributiveTaxation; max_iter=10e5, tol=1e-6,
                                    interpolation_method=interpolate_linear)
    @unpack a_grid, R, β, Π, σ, N, M = mdl
    # An intial guess is just linear
    policy_guess = reshape(repeat(a_grid, inner=(N,1)), N, M)
    # Update policy
    iter = 0
    diff = Inf
    while any(diff .> tol) && iter <= max_iter
        policy_previous, new_a_grid = update_policy_edogeneous(mdl, policy_guess, a_grid, interpolation_method=interpolation_method)
        diff = policy_previous .- policy_guess .|> abs
        # diff = a_grid .- new_a_grid .|> abs
        iter += 1
        print("Iteration: $(iter), diff = $(sum(diff))              \r")
        policy_guess, a_grid = policy_previous, new_a_grid
    end
    return policy_guess, a_grid
end
