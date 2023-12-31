#=
Custom structs and functions used in the other two scripts.
Structs are used to store parameters, environment variables, and observations.
=#

# -----------------------------------------------------------------------------
# 1. Package parameters, environment variables, and data into structs.
# -----------------------------------------------------------------------------
using Parameters

@with_kw mutable struct Model_Parameters_Struct{Type_1, Type_2, Type_3, Type_4, Type_5, Type_6}
    utility_parameters::Type_1
    true_quality::Type_2                            # true product quality
    Q₀::Type_3                                      # prior means
    σ²₀::Type_4                                     # prior variances
    σ²ₑ::Type_5                                     # experience signal noise (variance)
    σ²ₐ::Type_6                                     # experience signal noise (variance)
    #experience_signals::Array{Float64, 3}          # experience noise draws
    #advertising_signals::Array{Float64, 3}         # advertising noise draws
end

@with_kw mutable struct Environment_Struct
    n_individuals::Int64                            # model environment variables
    n_periods::Int64                                # ...
    n_choices::Int64
    n_simulations::Int64                            # parameters and draws for simulated likelihood
    experience_signals_simulated::Array{Float64, 4} # 
    advertising_signals_simulated::Array{Float64, 4}
end

@with_kw struct Data_Struct
    Choice_Count::Array{Int64, 3}
    Choice_Probabilities::Array{Float64, 3}
    Advertising::Array{Int64, 3}
    dₑ::Array{Int64, 3}                           
    dₐ::Array{Int64, 3}                           
    Nₑ::Array{Int64, 3}                           
    Nₐ::Array{Int64, 3}                           
end

# -----------------------------------------------------------------------------
# 2 Likelihood functions and other functions used in Bayesian MCMC.
# -----------------------------------------------------------------------------
# Calculate the likelihood of all observations for a given individual and given parameters.
function likelihood(Model_Parameters, Data, Environment)

    # Unpack relevant parameters.
    @unpack utility_parameters, true_quality, Q₀, σ²₀, σ²ₑ, σ²ₐ = Model_Parameters
    @unpack Choice_Count, Choice_Probabilities, Advertising, dₑ, dₐ, Nₑ, Nₐ = Data
    @unpack n_periods, n_choices, n_simulations, 
        experience_signals_simulated, advertising_signals_simulated = Environment
    variable_type = typeof(utility_parameters[1]) # typically Float64, but different type during automatic differntiation
    
    # Likelihood variables and preallocated arrays. 
    ℓ = zeros(variable_type, n_simulations)  
    exp_v = zeros(variable_type, n_choices)
    choice_probs = zeros(variable_type, n_choices)

    # Adjust signal draws to correspond to the given signal variance.
    experience_signals_all = experience_signals_simulated .* sqrt(σ²ₑ[1])
    advertising_signals_all = advertising_signals_simulated .* sqrt(σ²ₐ[1])

    for n in 1:n_simulations

        ℓₙ = zero(variable_type)
        experience_signals = @views experience_signals_all[:,:,:,n]
        advertising_signals = @views advertising_signals_all[:,:,:,n]

        # Loop over all choices in all periods.
        for i in 1:n_individuals
            for t in 1:n_periods

                # Calculate choice probabilities.
                fill!(exp_v, 0.0)

                for j in 1:n_choices

                    # Get current beliefs.
                    if t == 1 # if this is the first period, get the priors.
                        
                        Qᵢₜⱼ = deepcopy(Q₀[j])
                        σ²ᵢₜⱼ = deepcopy(σ²₀[j])
                    
                    else # otherwise, calculate updated beliefs based on experience signals.

                        # Sum of experience and detailing signals up to when product was chosen.
                        ∑ₜQᴱⱼₜdⱼₜ = 0.0
                        ∑ₜAⱼₜdⱼₜ = 0.0
                        for 𝒯 in 1:t-1
                            if dₑ[i,𝒯,j] == 1
                                ∑ₜQᴱⱼₜdⱼₜ +=  (true_quality[j] + experience_signals[i,𝒯,j])
                                ##∑ₜQᴱⱼₜdⱼₜ +=  (true_quality[j] + experience_signals[i,𝒯,j] / sqrt(Choice_Count[i,𝒯,j]))
                            end                                             # adjust signal strength for number of signals
                            if dₐ[i,𝒯,j] == 1
                                ∑ₜAⱼₜdⱼₜ += (true_quality[j] + advertising_signals[i,𝒯,j])
                                ##∑ₜAⱼₜdⱼₜ += (true_quality[j] + advertising_signals[i,𝒯,j] / sqrt(Advertising[i,𝒯+1,j]))
                            end                                             # adjust signal strength for number of signals
                        end
                        
                        # Calculate updated beliefs. Equations 17 and 18 in Ching/Erdem/Keane 2013.  
                        σ²ᵢₜⱼ = 1 / ((1 / σ²₀[j]) + (Nₑ[i,t-1,j] / σ²ₑ[1]) + (Nₐ[i,t-1,j] / σ²ₐ[1]))
                        Qᵢₜⱼ = (σ²ᵢₜⱼ / σ²ₑ[1] * ∑ₜQᴱⱼₜdⱼₜ) + (σ²ᵢₜⱼ / σ²ₐ[1] * ∑ₜAⱼₜdⱼₜ) + (σ²ᵢₜⱼ / σ²₀[j] * Q₀[j])
                    end

                    # Calculate indirect utility.
                    exp_v[j] = exp(EU(Qᵢₜⱼ, σ²ᵢₜⱼ, σ²ₑ, σ²ₐ, utility_parameters))
                end

                # Calculate CCPs.
                fill!(choice_probs, 0.0)
                ∑exp_vⱼ = sum(exp_v)
                for j in eachindex(choice_probs)
                    choice_probs[j] = exp_v[j] / ∑exp_vⱼ
                end

                # Record the log likelihood of all choices in this period for this individual.
                for j in 1:n_choices
                    ℓₙ += log(choice_probs[j]) * Choice_Count[i,t,j]
                    # (log likelihood) × (number of times j was chosen) 
                end
            end
        end
        ℓ[n] = ℓₙ
    end

    return mean(ℓ)
end

# Helper function to create deep copies of Model_Parameters_Struct. 
function copy_struct(Model_Parameters::Model_Parameters_Struct)

    @unpack utility_parameters, true_quality, Q₀, σ²₀, σ²ₑ, σ²ₐ,
        experience_signals, advertising_signals = Model_Parameters

    Model_Parameters_Struct(
        deepcopy(utility_parameters),
        deepcopy(true_quality),          
        deepcopy(Q₀),                    
        deepcopy(σ²₀),
        deepcopy(σ²ₑ),                    
        deepcopy(σ²ₐ))
end

# Helper function to fill all struct fields with the same value.
function fill_struct!(Model_Parameters, fill_value)
    for field in fieldnames(typeof(Model_Parameters))

        field_value = getfield(Model_Parameters, field)

        if field_value isa AbstractArray
            field_value .= fill_value
        else
            setfield!(Model_Parameters, field, fill_value)
        end
    end
end

# Convert a parameter struct into a vector of all the values containted in the struct.
function struct_to_vector(Model_Parameters)

    struct_values = [vec(getfield(Model_Parameters, field)) for field in fieldnames(typeof(Model_Parameters))]
    values_vector = vcat(struct_values...)

    return values_vector
end

# Convert from a vector of values to a parameters struct.
function vector_to_struct(vector)

    example_struct = Model_Parameters # use the global parameters struct to define fields and sizes
    next_vector_index = 1
    struct_fields = []

    # Fill in each field of the output struct.
    for field in fieldnames(typeof(example_struct))

        # Get the size of the given field.
        field_values = getfield(example_struct, field)
        field_length = size(field_values[:], 1)
        field_size = size(field_values)

        # Get the relevant values from the vector and reshape to fit the field.
        vector_values = vector[next_vector_index : next_vector_index + field_length - 1]

        # Reshape the vector of values and add to the list of struct fields.
        reshaped_vector = reshape(vector_values, field_size)
        push!(struct_fields, reshaped_vector)

        # Update the position of used data in the vector.
        next_vector_index = next_vector_index + field_length
    end

    # Create the new struct.
    new_struct = Model_Parameters_Struct(struct_fields...)
    return new_struct
end

# -----------------------------------------------------------------------------
# Functions used in maximum likelihood estimation.

# Likelihood function that takes a vector of parameter values as input.
function likelihood_objective(parameter_vector)

    # Put parameter values into a struct.
    Model_Parameters = vector_to_struct(parameter_vector)

    # Get the likelihood of this set of parameters.
    ℓ = likelihood(Model_Parameters, Data, Environment)

    return ℓ
end

function likelihood_objective(parameter_vector, true_values, free_parameter_indices)

    # Put free parameters and remaining fixed parameter values into a struct.
    input_vec = Vector{typeof(parameter_vector[1])}(struct_to_vector(true_values))
    input_vec[free_parameter_indices] = parameter_vector
    Model_Parameters = vector_to_struct(input_vec)

    # Get the likelihood of this set of parameters.
    ℓ = likelihood(Model_Parameters, Data, Environment)

    return ℓ
end

likelihood_objective(parameter_vector) = likelihood_objective(parameter_vector, true_values, free_parameter_indices)

# Likelihood gradient function using ForwardDiff automatic differentiation.
function likelihood_gradient(parameter_vector)
   return ForwardDiff.gradient(likelihood_objective, parameter_vector)
end

# Likelihood Hessian function using ForwardDiff automatic differentiation.
function likelihood_hessian(parameter_vector)
    return ForwardDiff.hessian(likelihood_objective, parameter_vector)
end

# Plot the maximum likelihood functions. 
function plot_likelihood_function(Model_Parameters)

    all_parameters = [:utility_parameters, :true_quality, :Q₀, :σ²₀, :σ²ₑ, :σ²ₐ]
    vector_index = 1

    for parameter_field in all_parameters
            
        parameter_values = getfield(Model_Parameters, parameter_field)

        for parameter_index in eachindex(parameter_values)

            true_value = parameter_values[parameter_index]
            x_range = LinRange(true_value - 1.0, true_value + 1.0, 20)
            if parameter_field ∈ [:σ²₀, :σ²ₑ, :σ²ₐ]
                x_range = LinRange(true_value * 0.3, true_value * 3.0, 50)
            end
            y = similar(x_range)

            for index in eachindex(x_range)
                parameter_values[parameter_index] = x_range[index]
                y[index] = likelihood(Model_Parameters, Data, Environment)

                # Alternatively, plot the derivative with respect to the given parameter.
                #parameter_vector = struct_to_vector(Model_Parameters)
                #y[index] = likelihood_gradient(parameter_vector)[vector_index]
            end

            fig = plot(x_range, y, title="parameter $parameter_field[$parameter_index], true value $true_value", label = "")
            plot!([true_value, true_value], [minimum(y), maximum(y)], linestyle = :dash, color = :black, label = "true value")
            display(fig)
            parameter_values[parameter_index] = true_value
            vector_index += 1
        end
    end
end