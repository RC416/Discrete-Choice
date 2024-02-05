using Parameters

@with_kw mutable struct Model_Parameters_Struct{Type1, Type2}
    β_mean::Type1 # ::Vector{Float64} except during automatic differentiation
    β_var::Type2  # ::Vector{Float64} except during automatic differentiation
end

@with_kw mutable struct Environment_Struct
    n_available_choices::Int64
    n_individuals::Int64
    n_individual_choices::Int64
    n_simulations::Int64
    β_draws::Array{Float64, 3}
end

@with_kw struct Data_Struct
    X::Array{Float64, 4}
    Choices::Array{Int64, 2}
end