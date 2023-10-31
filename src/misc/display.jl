function Base.show(::IO, mdl::EconomicsModel)
    println("")
    termshow(mdl)
end


function Base.show(::IO, sol::HouseholdFinanceSolution)
    println("Solution object for Athreya, Mustre-del Río, and Sánchez (2019)")
end
