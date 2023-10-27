export AbstractUtility, LogUtility, CRRAUtility, GHHUtility


abstract type AbstractUtility end


@doc raw"""Log Utility

Type used to evaluate log utility. Log utility takes the form

```math
u(c) = \log(c)
```
where `c` is the level of consumption.
Non-positive value for `c` are assigned an utility value of `-Inf`.

"""
struct LogUtility <: AbstractUtility end

(u::LogUtility)(c) = c > 0 ? log(c) : -Inf


@doc raw"""Constant relative risk-averse preferences"""
struct CRRAUtility <: AbstractUtility
    σ::Float64
end

(u::CRRAUtility)(c) = c > 0 ? c^(1 - u.σ)/(1 - u.σ) : -Inf


@doc raw"""Greenwood-Hercowitz-Huffman preferences"""
struct GHHUtility <: AbstractUtility
    θ::Float64
    γ::Float64
end


function (u::GHHUtility)(c, l)
    base = c - l^(1+u.θ)/(1+u.θ)
    return base > 0 ? base^(1-u.γ) / (1-u.γ) : -Inf
end
