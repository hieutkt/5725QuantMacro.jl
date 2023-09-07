export HP_filter


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
