export HouseholdFinance,
    HouseholdFinanceSolution,
    solve, compute_behavior


@kwdef struct HouseholdFinance <: EconomicsModel
    # PREFERENCES-RELATED PARAMETERS
    σ::Float64 = 2.0            # CRRA params
    u::AbstractUtility = CRRAUtility(σ)
    β::Float64 = 0.9            # Discount factor
    κ::Float64 = 0.05           # Type-1 EVD
    # LIFECYCLE COMPONENT OF INCOME PROCESS
    n_min::Int64 = 25           # Minimum age
    n_max::Int64 = 82           # Maximum age
    W::Int64 = 65               # Age of retirement
    α_a::Float64 = 0.096        # Linear term of the life-cycle function
    α_b::Float64 = -0.0022      # Quaratic term of the life-cycle function
    lifecycle_f::Function =     # Life-cycle function
        n -> log(1 + α_a * (n - 25) + α_b * (n - 25)^2)
    # PERSISTENT COMPONENT OF INCOME
    z_grid::Array{Float64} =    # Grid of z values
        [-0.1418, -0.0945, -0.0473, 0, 0.0473, 0.0945, 0.1418]
    Π::Matrix{Float64} =        # Transition matrix
        [0.9868  0.0132  0       0       0       0       0     ;
         0.007   0.9813  0.0117  0       0       0       0     ;
         0       0.008   0.9817  0.0103  0       0       0     ;
         0       0       0.0091  0.9818  0.0091  0       0     ;
         0       0       0       0.0103  0.9817  0.008   0     ;
         0       0       0       0       0.0117  0.9813  0.007 ;
         0       0       0       0       0       0.0132  0.9868]
    N::Int64 = length(z_grid) # The size of the z grid
    y_pension = 1.0
    # EXOGENEOUS COMPONENT OF INCOME
    ε_grid::Array{Float64} =    # Grid of ε values
        [-0.1, -0.05, 0., 0.05, 0.1]
    ε_prob::Array{Float64} =    # Probabilities of ε values
        [0.0668, 0.2417, 0.3829, 0.2417, 0.0668]
    L::Int64 = length(ε_grid)   # The size of ε values
    # DELIQUENCY AND DEFAULT PARAMETERS
    η::Float64 = 0.15           # Roll-over interest rate on deliquent debt
    τ_n::Function =             # Earning threshold in deliquency
     n -> 2.8*lifecycle_f(n)
    γ::Float64 = 0.35           # Discharge shock to deliquency debt
    f::Float64 = 0.12           # Bankrupcy filing cost
    # WEALTH-RELATED PARAMETERS
    r::Float64 = 0.03           # Risk-free interest rate
    a_min::Float64 = -0.5       # Mimimum wealth
    a_max::Float64 = 30.0       # Maximum wealth
    M::Int64 = 500              # The size of the wealth grid
    a_grid::Array{Float64} =   # The wealth grid
        LinRange(a_min, a_max, M)
end


###############################################################################
#                              Solving the model                              #
###############################################################################


@kwdef struct HouseholdFinanceSolution <: ModelSolution
    G::Dict{Int64, Array{Float64}} = Dict()
    V::Dict{Int64, Array{Float64}} = Dict()
    B::Dict{Int64, Array{Float64}} = Dict()
    D::Dict{Int64, Array{Float64}} = Dict()
    P_V::Dict{Int64, Array{Float64}} = Dict()
    P_B::Dict{Int64, Array{Float64}} = Dict()
    P_D::Dict{Int64, Array{Float64}} = Dict()
    A::Dict{Int64, Array{Int64}} = Dict()
    q::Dict{Int64, Array{Float64}} = Dict()
end


"""Initialize a solution object"""
function initialize_solution(mdl::HouseholdFinance)
    @unpack n_min, W, n_max, N, M, L = mdl
    sol = HouseholdFinanceSolution()
    for age in n_min:W-1
        sol.G[age] = zeros(Float64, M, N, L)
        sol.V[age] = zeros(Float64, M, N, L)
        sol.B[age] = zeros(Float64, M, N, L)
        sol.D[age] = zeros(Float64, M, N, L)
        sol.P_V[age] = zeros(Float64, M, N, L)
        sol.P_B[age] = zeros(Float64, M, N, L)
        sol.P_D[age] = zeros(Float64, M, N, L)
        sol.A[age] = zeros(Int64, M, N, L)
        sol.q[age] = zeros(Float64, M, N)
    end
    for age in [W]
        sol.G[age] = zeros(Float64, M, N, L)
        sol.V[age] = zeros(Float64, M, N, L)
        sol.B[age] = zeros(Float64, M, N, L)
        sol.P_V[age] = zeros(Float64, M, N, L)
        sol.P_B[age] = zeros(Float64, M, N, L)
        sol.P_D[age] = zeros(Float64, M, N, L)
        sol.A[age] = zeros(Int64, M, N, L)
        sol.q[age] = zeros(Float64, M, N)
    end
    for age in W+1:n_max
        sol.V[age] = zeros(Float64, M, N)
        sol.A[age] = zeros(Int64, M, N)
        sol.q[age] = zeros(Float64, M, N)
        sol.P_V[age] = ones(Float64, M, N)
        sol.P_B[age] = zeros(Float64, M, N)
        sol.P_D[age] = zeros(Float64, M, N)
    end
    return sol
end


"""Solve the entire model backwards"""
function solve(mdl::HouseholdFinance)
    @unpack n_min, n_max, W, M, N, L, lifecycle_f, y_pension, z_grid, ε_grid = mdl
    # Mutable objects like `Dict` are dangerous, so make sure to only use
    # them at the top level code
    sol = initialize_solution(mdl)

    # Innitialize the income grid
    y_grid = Dict()
    for age ∈ n_min:W
        y_grid[age] = [exp(lifecycle_f(age) + z_grid[j_z] + ε_grid[k_ε]) for j_z ∈ 1:N, k_ε ∈ 1:L]
    end
    y_grid[W+1] = [max(0.1 + 0.9 * exp(z), y_pension) for z ∈ z_grid]

    # Solve the model backwards
    # For the last age
    println("Solving the retired problem for age: $(n_max)")
    backsolve_retired!(sol, mdl, n_max, zeros(M,N), y_grid[W+1])
    # For the retired age
    for age ∈ n_max-1:-1:W+1
        println("Solving the retired problem for age: $(age)")
        backsolve_retired!(sol, mdl, age, sol.G[age+1], y_grid[W+1])
    end
    # For the transition age
    println("Solving the transition problem for age: $(W)")
    backsolve_transition!(sol, mdl, sol.G[W+1], y_grid[W])
    # For the working age
    for age ∈ (W-1):-1:25
        println("Solving the young problem for age: $(age)")
        backsolve_young!(sol, mdl, age, sol.G[age+1], y_grid[age],
                         sol.P_V[age+1], sol.P_D[age+1], sol.q[age+1])
    end
    return sol
end



"""This function returns the expectecd value and a triple-tuple containing
probabilities of choosing between paying debts, declaring bankrupt, and deliquency"""
function compute_behavior(mdl::HouseholdFinance, V::Float64, B::Float64, D::Float64)
    @unpack κ = mdl
    V_over_κ, B_over_κ, D_over_κ = V/κ,  B/κ, D/κ
    # Convert the ratio to big number format if it's apt
    exp_V_over_κ = abs(V_over_κ) > 700 ? exp(big(V_over_κ)) : exp(V_over_κ)
    exp_B_over_κ = abs(B_over_κ) > 700 ? exp(big(B_over_κ)) : exp(B_over_κ)
    exp_D_over_κ = abs(D_over_κ) > 700 ? exp(big(D_over_κ)) : exp(D_over_κ)
    # Compute the probability of choosing to paying debt, bankrupcy or deliquency
    composite_value = exp_V_over_κ + exp_B_over_κ + exp_D_over_κ
    𝔼_G = κ*log(composite_value) |> Float64
    prob_V = exp_V_over_κ / composite_value |> Float64
    prob_B = exp_B_over_κ / composite_value |> Float64
    prob_D = exp_D_over_κ / composite_value |> Float64
    # Returns the values characterize the agent's choice behavior
    return 𝔼_G, prob_V, prob_B, prob_D
end



"""Solve for the problem in retirement age"""
function backsolve_retired!(sol::HouseholdFinanceSolution, mdl::HouseholdFinance,
                            age::Int64, G′::Array{Float64, 2}, y_grid::Vector{Float64})
    @unpack u, β, n_max, M, N, W, z_grid, a_grid, r, τ_n, f, γ, η = mdl

    # Compute the restricted index for a since we're not allowing debts
    i_a₀ = searchsortedfirst(a_grid, 0.0)

    # Compute the prices of debt
    sol.q[age] = q = repeat([1/(1+r)], M, N)

    if age == n_max             # At maximum age, there is no cumulation
        sol.V[age] = u.([(1+r)*a_grid[i_a] + y_grid[j_z] for i_a in 1:M, j_z in 1:N])
        sol.A[age] .= i_a₀
    else
        # Main loop
        for i_a in 1:M, j_z in 1:N
            # Value for paying debts
            V_bellman = [u((1+r)*a_grid[i_a] + y_grid[j_z] - a_grid[i_a′]*q[i_a′, j_z]) + β*G′[i_a′, j_z] for i_a′ ∈ i_a₀:M]
            v, a = findmax(V_bellman)
            sol.V[age][i_a, j_z] = v
            sol.A[age][i_a, j_z] = a + i_a₀ - 1
        end
        sol.V[age][1:i_a₀-1,:] .= -Inf
    end
    # Value for G is the same as V
    sol.G[age] = sol.V[age]
end


"""Solve for the problem at transition age to retirement
We allow for Bankruptcy but not Deliquency.
Income transition (by matrix Π) doesn't apply in this period"""
function backsolve_transition!(sol::HouseholdFinanceSolution, mdl::HouseholdFinance, G′::Array{Float64, 2}, y_grid::Array{Float64, 2})
    @unpack u, β, W, n_max, L, M, N, z_grid, a_grid, r, f = mdl
    # Compute the prices of debt
    q = sol.q[W] = repeat([1/(1+r)], M, N)

    # Main loop
    for i_a in 1:M, j_z in 1:N, k_ε in 1:L
        # Value for paying debts
        V_bellman = [u((1+r)*a_grid[i_a] + y_grid[j_z, k_ε] - a_grid[i_a′]*q[i_a′, j_z]) + β*G′[i_a′, j_z] for i_a′ ∈ 1:M]
        V, sol.A[W][i_a, j_z, k_ε] = findmax(V_bellman)
        # Value for going bankrupt
        B = u(y_grid[j_z, k_ε] - f) + β*G′[i_a, j_z]
        # Compute the optimal behavior
        sol.G[W][i_a, j_z, k_ε], sol.P_V[W][i_a, j_z, k_ε], sol.P_B[W][i_a, j_z, k_ε], _ = compute_behavior(mdl, V, B, -Inf)
        sol.V[W][i_a, j_z, k_ε] = V
        sol.B[W][i_a, j_z, k_ε] = B
    end
end


"""Solve for the problem in working age"""
function backsolve_young!(sol::HouseholdFinanceSolution, mdl::HouseholdFinance, age, G′::Array{Float64, 3}, y_grid::Array{Float64, 2},
                         P_V′::Array{Float64, 3}, P_D′::Array{Float64, 3}, q′::Array{Float64,2})
    @unpack u, β, n_max, M, N, L, z_grid, a_grid, ε_prob, Π, r, τ_n, f, γ, η = mdl

    # Need to interpolate the prices the agent has to pay if they rollover
    q′_rollover = [interpolate_linear(a_grid[i_a′]*(1+η), a_grid, q′[:, j_z]) for i_a′ ∈ 1:M, j_z ∈ 1:N]
    # Compute the probability of paying back debts, either formally or informaly
    prob_pay = [P_V′[i_a′, j_z, k_ε] .+ P_D′[i_a′, j_z, k_ε]*(1-γ)*(1+η)*q′_rollover[i_a′, k_ε] for i_a′ ∈ 1:M, j_z ∈ 1:N, k_ε ∈ 1:L]
    prob_pay = [prob_pay[i_a′, j_z, :] ⋅ ε_prob for i_a′ ∈ 1:M, j_z ∈ 1:N] # normalize over the ε dimension
    # Compute the prices of debt
    q = sol.q[age] =  1/(1+r) * [Π[j_z, :] ⋅ prob_pay[i_a′, :] for i_a′ ∈ 1:M, j_z ∈ 1:N]

    # Need to interpolate for the future values of deliquency consequences
    G′_no_debt       = [interpolate_linear(0.0,               a_grid, G′[:, j_z, k_ε]) for   _ ∈ 1:M, j_z ∈ 1:N, k_ε ∈ 1:L]
    G′_rollover_debt = [interpolate_linear(a_grid[i_a]*(1+η), a_grid, G′[:, j_z, k_ε]) for i_a ∈ 1:M, j_z ∈ 1:N, k_ε ∈ 1:L]
    G′_deliquency    = (1-γ)*G′_no_debt .+ γ*G′_rollover_debt

    # Also: Precompute the expected future values
    𝔼_G′            = [ Π[j_z,:] ⋅ G′[i_a′, :, k_ε]           for i_a′ ∈ 1:M, j_z ∈ 1:N, k_ε ∈ 1:L] # 3D: M × N × L
    𝔼_G′_bankruptcy = [ Π[j_z,:] ⋅ G′_no_debt[1, :, k_ε]      for             j_z ∈ 1:N, k_ε ∈ 1:L] # 2D: N × L
    𝔼_G′_deliquency = [ Π[j_z,:] ⋅ G′_deliquency[i_a, :, k_ε] for i_a  ∈ 1:M, j_z ∈ 1:N, k_ε ∈ 1:L] # 3D: M × N × L

    # Main loop
    for i_a in 1:M, j_z in 1:N, k_ε in 1:L
        # Value for paying debts
        V_bellman = [u((1+r)*a_grid[i_a] + y_grid[j_z, k_ε] - a_grid[i_a′]*q[i_a′, j_z]) + β*𝔼_G′[i_a′, j_z, k_ε]
                     for i_a′ ∈ 1:M]
        V, sol.A[age][i_a, j_z, k_ε] = findmax(V_bellman)
        # Value for going bankrupt
        B = u(y_grid[j_z, k_ε] - f) + β*𝔼_G′_bankruptcy[j_z, k_ε]
        # Value for choosing deliquency
        D = u(min(y_grid[j_z], τ_n(age))) + β*𝔼_G′_deliquency[i_a, j_z, k_ε]
        # Compute the optimal behavior
        G, P_V, P_B, P_D = compute_behavior(mdl, V, B, D)
        sol.G[age][i_a, j_z, k_ε] = G
        sol.P_V[age][i_a, j_z, k_ε] = P_V
        sol.P_B[age][i_a, j_z, k_ε] = P_B
        sol.P_D[age][i_a, j_z, k_ε] = P_D
        sol.V[age][i_a, j_z, k_ε] = V
        sol.B[age][i_a, j_z, k_ε] = B
    end
end
