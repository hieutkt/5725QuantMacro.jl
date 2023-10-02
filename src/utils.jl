export HP_filter,
    tauchen_discretize, rouwenhorst_discretize,
    interpolate_linear


"""Hodrick–Prescott time series filter"""
function HP_filter(x::Vector, λ::Int)
    n = length(x)
    m = 2
    @assert n > m
    I = Diagonal(ones(n))
    D = spdiagm(0 => fill(1, n-m), -1 => fill(-2, n-m), -2 => fill(1, n-m) )
    @inbounds D = D[1:n,1:n-m]
    return (I + λ * D * D') \ x
end


@doc raw"""
The Tauchen method for generating a Markov transition matrix from an AR(1) process


# Example

```julia
# Parameters for the AR(1) process
σₗ = 0.4
ρ  = 0.6
σₑ = σₗ*sqrt(1- ρ^2)

# Generate the labor states
n=7
ln_labor_lower=-3*sqrt(σₗ)
ln_labor_upper=-1*ln_labor_lower
ln_labor_grid= LinRange(ln_labor_lower, ln_labor_upper, n)
grid_labor=exp.(ln_labor_grid)

# Using the Tauchen method
Π = tauchen_discretize(ln_labor_grid, ρ, σₑ)
```

"""
function tauchen_discretize(grid, ρ, σₑ)
    n = length(grid)
    Π = zeros(n, n)
    step_half = abs(grid[2] - grid[1]) / 2
    d = Normal()
    Π[:, 1]     = @.     cdf(d, (grid[1] - ρ*grid + step_half) / σₑ)
    Π[:, n]     = @. 1 - cdf(d, (grid[n] - ρ*grid - step_half) / σₑ)
    for j in 2:n-1
        Π[:, j] = @.     cdf(d, (grid[j] - ρ*grid + step_half) / σₑ) -
                         cdf(d, (grid[j] - ρ*grid - step_half) / σₑ)
    end
    return Π
end

@doc raw"""
The Rouwenhorst method for generating a Markov transition matrix from an AR(1) process
"""
function rouwenhorst_discretize(grid, ρ)
    n = length(grid)
    p = q = (1 + ρ)/2
    P = [p   1-p
         1-q  q]
    for size = 3:n
        z = zeros(Float64, size-1)
        P = p*[P z; z' 0] + (1-p)*[z P; 0 z'] +
            (1-q)*[z' 0; P z] + q*[0 z'; z P]
        P[2:size-1, :] ./= 2
    end
    return P
end


function interpolate_linear(x::Float64, xs::Array{Float64, 1}, fs::Array{Float64,1})
        a = x <= xs[1] ? 1 : x >= xs[end] ? length(xs)-1 : searchsortedlast(xs, x)
        b = a+1
        return fs[a] + (x - xs[a])*(fs[b] - fs[a])/(xs[b]-xs[a])
    end
end
