export HouseholdLifeCycle,
    backsolve

@kwdef struct HouseholdLifeCycle <: EconomicsModel
    # Preferences
    σ::Float64 = 2.                      # CRRA Parameters
    u::AbstractUtility = CRRAUtility(σ)
    β::Float64 = 0.935                  # Discount rate
    # Production
    R::Float64 = 1.01                    # Interest rate
    # Population
    T::Int64 = 65                      # Retirement age
    n_grid::Array{Int64} = 25:80         # Age grid
    # Debt market
    M::Integer             = 500        # Size of asset grid
    a_min::Float64         = -0.5       # Minimal value for asset grid
    a_max::Float64         = 50.0        # Maximum value of asset grid
    a_grid::Array{Float64} =            # asset states
        LinRange(a_min, a_max, M)
    # Deterministic part
    f::Function = t -> log(-1.0135 + 0.1086*t - 0.001122*t^2)
    σ_ε::Float64 = sqrt(0.05)
    ε_grid::Array{Float64} = [-σ_ε/sqrt(2), σ_ε/sqrt(2)]
    # Income process
    N::Int64 = 10
    ρ_z::Float64  = 0.995
    σ_z0::Float64 = sqrt(0.15)
    σ_η::Float64 = sqrt(0.01)
    σ_zₜ::Array = initialize_σ_z(σ_z0, σ_η, ρ_z, T-25+1)
    z_grid::Dict = initialize_z_grid(σ_zₜ, N)
    Πₜ::Dict = initialize_Π(z_grid, σ_zₜ, ρ_z)
end


function initialize_σ_z(σ_z0, σ_η, ρ_z, age_length)
    σ_zₜ = zeros(Float64, age_length)
    σ_zₜ[1] = sqrt(σ_z0)
    for i in 1:age_length-1
        σ_zₜ[i+1] = σ_zₜ[i] * ρ_z + σ_η
    end
    return σ_zₜ
end


function initialize_z_grid(σ_zₜ, N)
    z_grid = Dict()
    for age in 25:65
        z_grid[age] = LinRange(-3*σ_zₜ[age-24], 3*σ_zₜ[age-24], N)
    end
    return sort(z_grid)
end


function initialize_Π(z_grid, σ_zₜ, ρ_z)
    Πₜ = Dict()
    for age in 25:64
        Πₜ[age] = rouwenhorst_discretize_nonstationary(z_grid[age+1], ρ_z, σ_zₜ[age-23], σ_zₜ[age-24])
    end
    return sort(Πₜ)
end




function backsolve_policy_retired(mdl, v_future, y_grid)
    @unpack R, β, a_grid, ε_grid, u, N, M = mdl
    v_bellman = [u(R*a_grid[i_a] + y_grid[j_y] - a_grid[i_a′]) + β*v_future[i_a′, j_y]
                 for i_a ∈ 1:M, j_y ∈ 1:N, i_a′ ∈ 1:M]
    sol = map(findmax, eachslice(v_bellman, dims=(1,2)))
    return first.(sol), last.(sol)
end


function backsolve_policy_transition(mdl, age, v_future)
    @unpack R, β, a_grid, ε_grid, z_grid, u, N, M = mdl
    y_grid = exp.([mdl.f(age) + z_grid[age][j_y] + ε_grid[k_ε]
                   for j_y ∈ 1:N, k_ε ∈ 1:2])
    v_bellman = [u(R * a_grid[i_a] + y_grid[j_y, k_ε] - a_grid[i_a′]) + β*v_future[i_a′, j_y]
                 for i_a ∈ 1:M, j_y ∈ 1:N, k_ε ∈ 1:2, i_a′ ∈ 1:M]
    sol = map(findmax, eachslice(v_bellman, dims=(1,2,3)))
    return first.(sol), last.(sol)
end

function backsolve_policy_young(mdl, age, v_future)
    @unpack R, β, a_grid, ε_grid, z_grid, Πₜ, u, N, M = mdl
    y_grid = exp.([mdl.f(age) + z_grid[age][j_y] + ε_grid[k_ε]
                   for j_y ∈ 1:N, k_ε ∈ 1:2])
    expected_v_future = dropdims(mean(v_future, dims=3), dims=3)
    v_bellman = [u(R*a_grid[i_a] + y_grid[j_y, k_ε] - a_grid[i_a′]) + β*Πₜ[age][j_y,:]⋅expected_v_future[i_a′, :]
                 for i_a ∈ 1:M, j_y ∈ 1:N, k_ε ∈ 1:2, i_a′ ∈ 1:M]
    sol = map(findmax, eachslice(v_bellman, dims=(1,2,3)))
    return first.(sol), last.(sol)
end


function backsolve(mdl)
    @unpack R, β, a_grid, ε_grid, z_grid, u, N, M = mdl
    y_retired_grid = log(0.7) .+ z_grid[65] .+ mdl.f(65) .|> exp
    v = Dict()
    g = Dict()
    v[80] = [u(R * a_grid[i_a] + y_retired_grid[j_y]) for i_a ∈ 1:M, j_y ∈ 1:N]
    for age in 79:-1:66
        println("Solving the retired problem for age: $(age)")
        v[age], g[age] = backsolve_policy_retired(mdl, v[age+1], y_retired_grid)
    end
    println("Solving the transition problem for age: 65")
    v[65], g[65] = backsolve_policy_transition(mdl, 65, v[66])
    for age in 64:-1:25
        println("Solving the young problem for age: $(age)")
        v[age], g[age] = backsolve_policy_young(mdl, age, v[age+1])
    end
    return sort(v), sort(g)
end
