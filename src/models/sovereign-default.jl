export SovereignDefault, SovereignDefaultWithoutShocks, SovereignDefaultWithInterpolation,
    # u,
    # Vᴳ_T, Vᴳ_prev,
    # Vᴰ_T, Vᴰ_prev,
    # V_T, V_Tm1, V_t,
    # D_T, D_t, q_Tm1, q_prev,
    # G_prevj′,
    # expected_value, expected_price,
    value_iterate_backwards


@kwdef struct SovereignDefault <: EconomicsModel
    # Preferences parameters
    γ::Float64 = 2.             # Relative risk aversion
    θ::Float64 = 0.5            # Elasticity of labor disutility
    β::Float64 = 0.89           # Discount rate
    ϕ::Float64 = 0.92           # Default threshold
    # Production parameters
    N_A::Int64 = 61             # Grid size for A
    ρ::Float64 = 0.9            # Autocorrelation
    σ_ϵ::Float64 = 0.01         # Standard deviation of productivity shock
    σ_A::Float64 = σ_ϵ/sqrt(1 - ρ^2) # Std. of labor productivity
    log_A_grid::Array{Float64} = LinRange(-3*σ_A, 3*σ_A, N_A)
    A_grid::Array{Float64} = exp.(log_A_grid)
    Π::Matrix{Float64}     =    # Transition matrix
        tauchen_discretize(log_A_grid, ρ, σ_ϵ)
    # Shock parameters
    κ::Float64   = 0.03         # EVD parameter for Government
    κ_η::Float64 = 0.001
    # Borrowing grid
    N_b::Int64 = 61             # Grid size for b
    b_grid = LinRange(-0.3, 0, N_b)  # Borrowing grid
    r::Float64 = 0.03           # World interest rate
end


"""
    u(mdl::SovereignDefault, c, ℓ)

GHH felicity function
"""
function u(mdl::SovereignDefault, c, ℓ)
    @unpack γ, θ = mdl
    base = (c - ℓ^(1+θ)/(1+θ))
    return base <= 0 ? -Inf : base^(1-γ)/(1-γ)
end


"""
Value of the government in period T if choose not to default
"""
function Vᴳ_T(mdl::SovereignDefault, A, b)
    @unpack κ_η, N_b, θ = mdl
    return κ_η * log(N_b) + u(mdl, A^(1+1/θ) + b, A^(1/θ))
end


"""
Backsolving for Vᴳ at time t < T
"""
function Vᴳ_prev(mdl::SovereignDefault, A, b, V_guess, q_mat)
    @unpack β, κ_η, θ, ϕ, Π, A_grid, b_grid, N_b = mdl
    A_idx = searchsortedfirst(A_grid, A)
    v_now = [u(mdl, A^(1+1/θ) + b - q_mat[A_idx, searchsortedfirst(b_grid, b′)] * b′, A^(1/θ)) for b′ ∈ b_grid]
    v_future = [β * Π[A_idx, :] ⋅ V_guess[:, b′_idx] for b′_idx ∈ 1:N_b]
    v_over_κ = (v_now .+ v_future) ./ κ_η .|> big
    return κ_η * log(sum(exp.(v_over_κ)))
end


"""
Value of the government in period T if choose to default
"""
function Vᴰ_T(mdl::SovereignDefault, A)
    @unpack κ_η, θ, ϕ = mdl
    return u(mdl, min(A, ϕ)^(1+1/θ), A^(1/θ))
end


"""
Backsolving for Vᴰ given a matrix of future values
"""
function Vᴰ_prev(mdl::SovereignDefault, A, V_guess)
    @unpack β, κ_η, θ, ϕ, Π, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b′_idx = searchsortedfirst(b_grid, 0.0)
    v_now = u(mdl, min(A, ϕ)^(1+1/θ), A^(1/θ))
    v_future = β * Π[A_idx, :] ⋅ V_guess[:, b′_idx]
    return v_now + v_future
end


"""
Probability to choose default in period T
"""
function D_T(mdl::SovereignDefault, A, b)
    @unpack κ = mdl
    Vᴳ_over_κ, Vᴰ_over_κ = Vᴳ_T(mdl, A, b)/κ, Vᴰ_T(mdl, A)/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return exp(Vᴰ_over_κ)/(exp(Vᴰ_over_κ) + exp(Vᴳ_over_κ)) |> Float64
end


"""
Probability to choose default in any period, given Vᴳ and Vᴰ
"""
function D_t(mdl::SovereignDefault, A, b, Vᴳ, Vᴰ)
    @unpack κ, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b_idx = searchsortedfirst(b_grid, b)
    Vᴳ_over_κ, Vᴰ_over_κ = Vᴳ[A_idx, b_idx]/κ, Vᴰ[A_idx]/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return exp(Vᴰ_over_κ)/(exp(Vᴰ_over_κ) + exp(Vᴳ_over_κ)) |> Float64
end



"""
Value function in period T
"""
function V_T(mdl::SovereignDefault, A, b)
    @unpack κ = mdl
    Vᴳ_over_κ = Vᴳ_T(mdl, A, b)/κ
    Vᴰ_over_κ = Vᴰ_T(mdl, A)/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return κ * log(exp(Vᴳ_over_κ) + exp(Vᴰ_over_κ)) |> Float64
end


"""
Backsolving for value function (for t = T - 1)
"""
function V_Tm1(mdl::SovereignDefault, A, b, Vᴳ, Vᴰ)
    @unpack κ, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b_idx = searchsortedfirst(b_grid, b)
    Vᴳ_over_κ, Vᴰ_over_κ = Vᴳ[A_idx, b_idx]/κ, Vᴰ[A_idx]/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return κ * log(exp(Vᴳ_over_κ) + exp(Vᴰ_over_κ)) |> Float64
end


"""
Backsolving for value function (for t < T - 1)
"""
function V_t(mdl::SovereignDefault, A, b, Vᴳ, Vᴰ)
    @unpack κ, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b_idx = searchsortedfirst(b_grid, b)
    Vᴳ_over_κ, Vᴰ_over_κ = Vᴳ[A_idx, b_idx]/κ, Vᴰ[A_idx]/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return κ * log(exp(Vᴳ_over_κ) + exp(Vᴰ_over_κ)) |> Float64
end


"""
Back-solving debt choice, without future value (implied t = T-1)
"""
function q_Tm1(mdl::SovereignDefault, A, b′)
    @unpack Π, A_grid, r = mdl
    A_idx = searchsortedfirst(A_grid, A)
    D_Ts = map(A′ -> D_T(mdl, A′, b′), A_grid)
    return ( Π[A_idx,:] ⋅ (1 .- D_Ts) ) / (1 + r)
end


"""
Back-solving debt choice, without future value (implied t < T-1)
"""
function q_prev(mdl::SovereignDefault, A, b′, Vᴳ, Vᴰ)
    @unpack Π, A_grid, r = mdl
    A_idx = searchsortedfirst(A_grid, A)
    D_ts = map(A′ -> D_t(mdl, A′, b′, Vᴳ, Vᴰ), A_grid)
    return ( Π[A_idx,:] ⋅ (1 .- D_ts) ) / (1 + r)
end



"""Compute the probability that we we choose the grid point b′ in the future.
This function returns a vector of conditional probability"""
function G_prevj′(mdl::SovereignDefault, A, b, V, q)
    @unpack β, Π, A_grid, b_grid, θ, N_b = mdl
    A_idx = searchsortedfirst(A_grid, A)
    v_futures = [β * Π[A_idx, :] ⋅ V[:, b′_idx] for b′_idx ∈ 1:N_b]
    v_now = [u(mdl, A^(1 + 1/θ) + b - q[A_idx, b′_idx]*b_grid[b′_idx], A^(1/θ)) for b′_idx ∈ 1:N_b]
    v = exp.(v_futures .+ v_now)
    return v./sum(v)
end


###############################################################################
#                             Model without shocks                            #
###############################################################################

@kwdef struct SovereignDefaultWithoutShocks <: EconomicsModel
    # Preferences parameters
    γ::Float64 = 2.             # Relative risk aversion
    θ::Float64 = 0.5            # Elasticity of labor disutility
    β::Float64 = 0.89           # Discount rate
    ϕ::Float64 = 0.92           # Default threshold
    # Production parameters
    N_A::Int64 = 61             # Grid size for A
    ρ::Float64 = 0.9            # Autocorrelation
    σ_ϵ::Float64 = 0.01         # Standard deviation of productivity shock
    σ_A::Float64 = σ_ϵ/sqrt(1 - ρ^2) # Std. of labor productivity
    log_A_grid::Array{Float64} = LinRange(-3*σ_A, 3*σ_A, N_A)
    A_grid::Array{Float64} = exp.(log_A_grid)
    Π::Matrix{Float64}     =    # Transition matrix
        tauchen_discretize(log_A_grid, ρ, σ_ϵ)
    # Borrowing grid
    N_b::Int64 = 61             # Grid size for b
    b_grid = LinRange(-0.3, 0, N_b)  # Borrowing grid
    r::Float64 = 0.03           # World interest rate
end


"""
    u(mdl::SovereignDefault, c, ℓ)

GHH felicity function
"""
function u(mdl::SovereignDefaultWithoutShocks, c, ℓ)
    @unpack γ, θ = mdl
    base = (c - ℓ^(1+θ)/(1+θ))
    return base <= 0 ? -Inf : base^(1-γ)/(1-γ)
end


"""
Value of the government in period T if choose not to default
"""
function Vᴳ_T(mdl::SovereignDefaultWithoutShocks, A, b)
    @unpack N_b, θ = mdl
    return u(mdl, A^(1+1/θ) + b, A^(1/θ))
end


"""
Backsolving for Vᴳ at time t < T-1
"""
function Vᴳ_prev(mdl::SovereignDefaultWithoutShocks, A, b, V_guess, q_mat)
    @unpack β, θ, ϕ, Π, A_grid, b_grid, N_b = mdl
    A_idx = searchsortedfirst(A_grid, A)
    v_now = [u(mdl, A^(1+1/θ) + b - q_mat[A_idx, b′_idx] * b_grid[b′_idx], A^(1/θ)) for b′_idx ∈ 1:N_b]
    v_future = [β * Π[A_idx, :] ⋅ V_guess[:, b′_idx] for b′_idx ∈ 1:N_b]
    return maximum(v_now .+ v_future)
end


"""
Value of the government in period T if choose to default
"""
function Vᴰ_T(mdl::SovereignDefaultWithoutShocks, A)
    @unpack θ, ϕ = mdl
    return u(mdl, min(A, ϕ)^(1+1/θ), A^(1/θ))
end


"""
Backsolving for Vᴰ given a matrix of future values
"""
function Vᴰ_prev(mdl::SovereignDefaultWithoutShocks, A, V_guess)
    @unpack β, θ, ϕ, Π, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b′_idx = searchsortedfirst(b_grid, 0.0)
    v_now = u(mdl, min(A, ϕ)^(1+1/θ), A^(1/θ))
    v_future = β * Π[A_idx, :] ⋅ V_guess[:, b′_idx]
    return v_now + v_future
end



# Check
"""
Probability to choose default in any period
"""
function D_t(mdl::SovereignDefaultWithoutShocks, A, b)
    @unpack θ, ϕ = mdl
    return A^(1+1/θ) + b < min(A, ϕ)^(1+1/θ)
end

D_T(mdl::SovereignDefaultWithoutShocks, A, b) = D_t(mdl::SovereignDefaultWithoutShocks, A, b)


"""
Value function in period 2
"""
function V_T(mdl::SovereignDefaultWithoutShocks, A, b)
    @unpack θ, ϕ = mdl
    return u(mdl, max(A^(1+1/θ) + b, min(A, ϕ)^(1+1/θ)), A^(1/θ))
end


"""
Backsolving for value function (for t < T)
"""
function V_Tm1(mdl::SovereignDefaultWithoutShocks, A, b, Vᴳ_mat, Vᴰ_vec)
    return V_t(mdl::SovereignDefaultWithoutShocks, A, b, Vᴳ_mat, Vᴰ_vec)
end


"""
Backsolving for value function (for t < T - 1)
"""
function V_t(mdl::SovereignDefaultWithoutShocks, A, b, Vᴳ_mat, Vᴰ_vec)
    @unpack A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b_idx = searchsortedfirst(b_grid, b)
    Vᴳ, Vᴰ = Vᴳ_mat[A_idx, b_idx], Vᴰ_vec[A_idx]
    return max(Vᴳ, Vᴰ)
end

"""
Back-solving debt choice, without future value (implied t = T-1)
"""
function q_Tm1(mdl::SovereignDefaultWithoutShocks, A, b′)
    @unpack Π, A_grid, r = mdl
    A_idx = searchsortedfirst(A_grid, A)
    D_Ts = map(A′ -> D_T(mdl, A′, b′), A_grid)
    return ( Π[A_idx,:] ⋅ (1 .- D_Ts) ) / (1 + r)
end

q_prev(mdl::SovereignDefaultWithoutShocks, A, b′, Vᴳ, Vᴰ) = q_Tm1(mdl::SovereignDefaultWithoutShocks, A, b′)


"""Compute the probability that we we choose the grid point b′ in the future.
This function returns a vector of conditional probability"""
function G_prevj′(mdl::SovereignDefaultWithoutShocks, A, b, V, q)
    @unpack β, Π, A_grid, b_grid, θ, N_b = mdl
    A_idx = searchsortedfirst(A_grid, A)
    v_futures = [β * Π[A_idx, :] ⋅ V[:, b′_idx] for b′_idx ∈ 1:N_b]
    v_now = [u(mdl, A^(1 + 1/θ) + b - q[A_idx, b′_idx]*b_grid[b′_idx], A^(1/θ)) for b′_idx ∈ 1:N_b]
    v = v_futures .+ v_now
    return v .== maximum(v)
end


###############################################################################
#                                   Solver                                     #
###############################################################################

"""Solve the model by value interating backwards"""
function value_iterate_backwards(mdl; tol = 1e-7, max_iter = 1e5)
    @unpack A_grid, b_grid, N_A, N_b = mdl
    # Solve for value in time T
    V = [V_T(mdl, A, b) for A ∈ A_grid, b ∈ b_grid]
    # Solve for value in time T-1
    q  = [q_Tm1(mdl, A, b) for A ∈ A_grid, b ∈ b_grid]
    Vᴳ = [Vᴳ_prev(mdl, A, b, V, q) for A ∈ A_grid, b ∈ b_grid]
    Vᴰ = [Vᴰ_prev(mdl, A, V) for A ∈ A_grid]
    V_new = [V_Tm1(mdl, A, b, Vᴳ, Vᴰ) for A ∈ A_grid, b ∈ b_grid]
    # Loop: solve for value in back in time until convergence
    diff =  V_new .- V
    iter = 2
    while any(abs.(diff) .>= tol) && iter <= max_iter
        println("Iteration $iter, diff is $(sum(diff))")
        V = V_new
        q  = [q_prev(mdl, A, b, Vᴳ, Vᴰ) for A ∈ A_grid, b ∈ b_grid]
        Vᴳ = [Vᴳ_prev(mdl, A, b, V, q) for A ∈ A_grid, b ∈ b_grid]
        Vᴰ = [Vᴰ_prev(mdl, A, V) for A ∈ A_grid]
        V_new = [V_t(mdl, A, b, Vᴳ, Vᴰ) for A ∈ A_grid, b ∈ b_grid]
        diff =  V_new .- V
        iter += 1
    end
    return V, Vᴳ, Vᴰ
end


@kwdef struct SovereignDefaultWithInterpolation <: EconomicsModel
    # Preferences parameters
    γ::Float64 = 2.             # Relative risk aversion
    θ::Float64 = 0.5            # Elasticity of labor disutility
    β::Float64 = 0.89           # Discount rate
    ϕ::Float64 = 0.92           # Default threshold
    # Production parameters
    N_A::Int64 = 61             # Grid size for A
    ρ::Float64 = 0.9            # Autocorrelation
    σ_ϵ::Float64 = 0.01         # Standard deviation of productivity shock
    σ_A::Float64 = σ_ϵ/sqrt(1 - ρ^2) # Std. of labor productivity
    log_A_grid::Array{Float64} = LinRange(-3*σ_A, 3*σ_A, N_A)
    A_grid::Array{Float64} = exp.(log_A_grid)
    Π::Matrix{Float64}     =    # Transition matrix
        tauchen_discretize(log_A_grid, ρ, σ_ϵ)
    # Interpolation grid for A
    N_Â = (N_A - 1)*11 + 1
    log_Â_grid::Array{Float64} = LinRange(-3*σ_A, 3*σ_A, N_Â)
    Â_grid::Array{Float64} = exp.(log_Â_grid)
    Π̂::Matrix{Float64}     =    # Transition matrix
        tauchen_discretize(log_Â_grid, ρ, σ_ϵ)
    # Shock parameters
    κ::Float64   = 0.03         # EVD parameter for Government
    κ_η::Float64 = 0.001
    # Borrowing grid
    N_b::Int64 = 61             # Grid size for b
    b_grid = LinRange(-0.3, 0, N_b)  # Borrowing grid
    r::Float64 = 0.03           # World interest rate
end


function expected_value(mdl, A, b′, V;
                        interpolation_method = interpolate_linear)
    @unpack β, Π̂, A_grid, Â_grid, b_grid, N_b = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b′_idx = searchsortedfirst(b_grid, b′)
    return Π̂[A_idx + 10*(A_idx-1), :] ⋅ map(Â -> interpolation_method(Â, A_grid, V[:, b′_idx]), Â_grid)
end


function expected_price(mdl, A, b′, D;
                        interpolation_method = interpolate_linear)
    @unpack β, Π̂, A_grid, Â_grid, b_grid, N_b = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b′_idx = searchsortedfirst(b_grid, b′)
    return Π̂[A_idx + 10*(A_idx-1), :] ⋅ (1 .- map(Â -> interpolation_method(Â, A_grid, D[:, b′_idx]), Â_grid))
end


"""
    u(mdl::SovereignDefault, c, ℓ)

GHH felicity function
"""
function u(mdl::SovereignDefaultWithInterpolation, c, ℓ)
    @unpack γ, θ = mdl
    base = (c - ℓ^(1+θ)/(1+θ))
    return base <= 0 ? -Inf : base^(1-γ)/(1-γ)
end


"""
Value of the government in period T if choose not to default
"""
function Vᴳ_T(mdl::SovereignDefaultWithInterpolation, A, b)
    @unpack κ_η, N_b, θ = mdl
    return κ_η * log(N_b) + u(mdl, A^(1+1/θ) + b, A^(1/θ))
end


"""
Backsolving for Vᴳ at time t < T-1
"""
function Vᴳ_prev(mdl::SovereignDefaultWithInterpolation, A, b, 𝔼V, q_mat)
    @unpack β, κ_η, θ, ϕ, Π, A_grid, b_grid, N_b = mdl
    A_idx = searchsortedfirst(A_grid, A)
    v_now = [u(mdl, A^(1+1/θ) + b - q_mat[A_idx, searchsortedfirst(b_grid, b′)] * b′, A^(1/θ)) for b′ ∈ b_grid]
    v_future = β * 𝔼V[A_idx, :]
    v_over_κ = (v_now .+ v_future) ./ κ_η .|> big
    return κ_η * log(sum(exp.(v_over_κ)))
end


"""
Value of the government in period T if choose to default
"""
function Vᴰ_T(mdl::SovereignDefaultWithInterpolation, A)
    @unpack κ_η, θ, ϕ = mdl
    return u(mdl, min(A, ϕ)^(1+1/θ), A^(1/θ))
end


"""
Backsolving for Vᴰ given a matrix of future values
"""
function Vᴰ_prev(mdl::SovereignDefaultWithInterpolation, A, 𝔼V)
    @unpack β, κ_η, θ, ϕ, Π, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b_idx = searchsortedfirst(b_grid, 0.0)
    v_now = u(mdl, min(A, ϕ)^(1+1/θ), A^(1/θ))
    v_future = β * 𝔼V[A_idx, b_idx]
    return v_now + v_future
end


"""
Probability to choose default in period T
"""
function D_T(mdl::SovereignDefaultWithInterpolation, A, b)
    @unpack κ = mdl
    Vᴳ_over_κ, Vᴰ_over_κ = Vᴳ_T(mdl, A, b)/κ, Vᴰ_T(mdl, A)/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return exp(Vᴰ_over_κ)/(exp(Vᴰ_over_κ) + exp(Vᴳ_over_κ)) |> Float64
end


"""
Probability to choose default in any period, given Vᴳ and Vᴰ
"""
function D_t(mdl::SovereignDefaultWithInterpolation, A, b, Vᴳ, Vᴰ)
    @unpack κ, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b_idx = searchsortedfirst(b_grid, b)
    Vᴳ_over_κ, Vᴰ_over_κ = Vᴳ[A_idx, b_idx]/κ, Vᴰ[A_idx]/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return exp(Vᴰ_over_κ)/(exp(Vᴰ_over_κ) + exp(Vᴳ_over_κ)) |> Float64
end


"""
Back-solving debt choice, without future value (implied t = T-1)
"""
function q_Tm1(mdl::SovereignDefaultWithInterpolation, A, b′, D_T)
    @unpack Π, A_grid, r = mdl
    A_idx = searchsortedfirst(A_grid, A)
    return ( Π[A_idx,:] ⋅ (1 .- D_T[A_idx, :]) ) / (1 + r)
end


"""
Value function in period T
"""
function V_T(mdl::SovereignDefaultWithInterpolation, A, b)
    @unpack κ = mdl
    Vᴳ_over_κ = Vᴳ_T(mdl, A, b)/κ
    Vᴰ_over_κ = Vᴰ_T(mdl, A)/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return κ * log(exp(Vᴳ_over_κ) + exp(Vᴰ_over_κ)) |> Float64
end


"""
Backsolving for value function (for t = T - 1)
"""
function V_Tm1(mdl::SovereignDefaultWithInterpolation, A, b, Vᴳ, Vᴰ)
    @unpack κ, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b_idx = searchsortedfirst(b_grid, b)
    Vᴳ_over_κ, Vᴰ_over_κ = Vᴳ[A_idx, b_idx]/κ, Vᴰ[A_idx]/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return κ * log(exp(Vᴳ_over_κ) + exp(Vᴰ_over_κ)) |> Float64
end


"""
Backsolving for value function (for t < T - 1)
"""
function V_t(mdl::SovereignDefaultWithInterpolation, A, b, Vᴳ, Vᴰ)
    @unpack κ, A_grid, b_grid = mdl
    A_idx = searchsortedfirst(A_grid, A)
    b_idx = searchsortedfirst(b_grid, b)
    Vᴳ_over_κ, Vᴰ_over_κ = Vᴳ[A_idx, b_idx]/κ, Vᴰ[A_idx]/κ
    Vᴳ_over_κ = abs(Vᴳ_over_κ) > 700 ? big(Vᴳ_over_κ) : Vᴳ_over_κ
    Vᴰ_over_κ = abs(Vᴰ_over_κ) > 700 ? big(Vᴰ_over_κ) : Vᴰ_over_κ
    return κ * log(exp(Vᴳ_over_κ) + exp(Vᴰ_over_κ)) |> Float64
end


"""Compute the probability that we we choose the grid point b′ in the future.
This function returns a vector of conditional probability"""
function G_prevj′(mdl::SovereignDefaultWithInterpolation, A, b, V, q)
    @unpack β, Π, A_grid, b_grid, θ, N_b = mdl
    A_idx = searchsortedfirst(A_grid, A)
    v_futures = [β * Π[A_idx, :] ⋅ V[:, b′_idx] for b′_idx ∈ 1:N_b]
    v_now = [u(mdl, A^(1 + 1/θ) + b - q[A_idx, b′_idx]*b_grid[b′_idx], A^(1/θ)) for b′_idx ∈ 1:N_b]
    v = exp.(v_futures .+ v_now)
    return v./sum(v)
end


"""Solve the model by value interating backwards"""
function value_iterate_backwards(mdl::SovereignDefaultWithInterpolation;
                                 tol = 1e-7, max_iter = 1e5,
                                 interpolation_method = interpolate_linear)
    @unpack A_grid, b_grid, N_A, N_b, r = mdl
    # Solve for V and D at the last period
    V = [V_T(mdl, A, b) for A ∈ A_grid, b ∈ b_grid]
    D = [D_T(mdl, A, b) for A ∈ A_grid, b ∈ b_grid]
    # Solve for expected value and price
    𝔼_V = [expected_value(mdl, A, b, V, interpolation_method=interpolation_method)
           for A ∈ A_grid, b ∈ b_grid]
    𝔼_q = [expected_price(mdl, A, b, D, interpolation_method=interpolation_method)
           for A ∈ A_grid, b ∈ b_grid]
    # 4) Using E, solve for q
    q = 𝔼_q./(1+r)
    # Solve for value in time T-1
    Vᴳ = [Vᴳ_prev(mdl, A, b, 𝔼_V, q) for A ∈ A_grid, b ∈ b_grid]
    Vᴰ = [Vᴰ_prev(mdl, A, 𝔼_V) for A ∈ A_grid]
    V_new = [V_Tm1(mdl, A, b, Vᴳ, Vᴰ) for A ∈ A_grid, b ∈ b_grid]
    # Loop: solve for value in back in time until convergence
    diff =  V_new .- V
    iter = 2
    while any(abs.(diff) .>= tol) && iter <= max_iter
        println("Iteration $iter, diff is $(sum(diff))")
        V = V_new
        # Solve for V and D at the last period
        D = [D_t(mdl, A, b, Vᴳ, Vᴰ) for A ∈ A_grid, b ∈ b_grid]
        # Solve for expected value and price
        𝔼_V = [expected_value(mdl, A, b, V, interpolation_method=interpolation_method)
               for A ∈ A_grid, b ∈ b_grid]
        𝔼_q = [expected_price(mdl, A, b, D, interpolation_method=interpolation_method)
               for A ∈ A_grid, b ∈ b_grid]
        # 4) Using E, solve for q
        q = 𝔼_q./(1+r)
        # Solve for value in time T-1
        Vᴳ = [Vᴳ_prev(mdl, A, b, 𝔼_V, q) for A ∈ A_grid, b ∈ b_grid]
        Vᴰ = [Vᴰ_prev(mdl, A, 𝔼_V) for A ∈ A_grid]
        V_new = [V_t(mdl, A, b, Vᴳ, Vᴰ) for A ∈ A_grid, b ∈ b_grid]
        diff =  V_new .- V
        iter += 1
    end
    return V, Vᴳ, Vᴰ
end


