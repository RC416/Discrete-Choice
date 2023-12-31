#=
Limited consideration model.
See documentation.
=#

using Distributions, Parameters, StatsBase, DataFrames, Plots, BenchmarkTools,
    LinearAlgebra, Statistics, ForwardDiff

# -----------------------------------------------------------------------------
# 1.0 - Model Setup and Simulate Data.
# -----------------------------------------------------------------------------
# Environment.
n_individuals = 1000
n_periods = 10
n_products = 3

# Utility parameters.
utility_parameters = [0.5, 1.0]  # indicator for product, third choice has normalized utility of 0.0
u(αᵢⱼ) = αᵢⱼ   # utility function

# Choice set parameters.
choice_set_parameters = [1.0] # impact of choice set observables on choice set probability
ϕ(Xᵢⱼₜ, choice_set_parameters) = exp(Xᵢⱼₜ * choice_set_parameters[1]) / (
                    1.0 + exp(Xᵢⱼₜ *choice_set_parameters[1])) # choice set formation function

# Choice sets.
all_choice_sets =  [[0,0,0],
                    [0,0,1], [0,1,0], [1,0,0],
                    [0,1,1], [1,0,1], [1,1,0],
                    [1,1,1]]

# Observable: variable that affects choice set probabilities.
Choice_Set_Observable = rand(Normal(0,1), (n_individuals, n_periods, n_products))

# Draw true individual choice sets.
choice_sets = Array{Int64, 2}(undef, n_individuals, n_periods) # store the index of choice set for each i, t. 

for i in 1:n_individuals
    for t in 1:n_periods

        choice_set_probs = Vector{Float64}(undef, size(all_choice_sets,1))

        # Calculate the choice set probabilities. See Goeree 2008.
        for choice_set_index in eachindex(all_choice_sets)
            
            choice_set = all_choice_sets[choice_set_index]
            choice_probs = zeros(size(choice_set))

            # Calculate probability of each product being in the choice set.
            for j in 1:n_products               
                choice_probs[j] = ϕ(Choice_Set_Observable[i,t,j], choice_set_parameters)
            end

            # Calculate the likelihood of this choice set.
            choice_set_probability = 1.0
            for j in eachindex(choice_probs)

                # Check if the product is in the choice set.
                if choice_set[j] == 1
                    choice_set_probability *= choice_probs[j]
                else
                    choice_set_probability *= 1 - choice_probs[j]
                end
            end
            choice_set_probs[choice_set_index] = choice_set_probability
        end
    
        # Select choice set according to probabilities.
        choice_sets[i,t] = sample(eachindex(all_choice_sets), Weights(choice_set_probs))
    end
end

# Observable: prescription count and choice frequencies.
Choice_Count = Array{Int64, 2}(undef, n_individuals, n_periods)
Choice_Frequency = Array{Float64, 3}(undef, n_individuals, n_periods, n_products)

# Draw prescription counts.
for i in 1:n_individuals
    for t in 1:n_periods
        if choice_sets[i,t] == 1            # empty choice set
            Choice_Count[i,t] = 0
        else                                # non-empty choice set
            Choice_Count[i,t] = rand(Geometric(1/200))
        end
    end
end

# Calculate choice frequencies.
for i in 1:n_individuals
    for t in 1:n_periods

        # Terms in the logit CCP expression.
        exp_uⱼ = zeros(n_products)
        ∑exp_uⱼ = 0.0

        # Calculate utility for each choice.
        for j in 1:n_products

            # Get utility parameters.
            if j == n_products
                αⱼ = 0.0 # normalize utility for third product
            else
                αⱼ = utility_parameters[j]
            end

            # Get utility for this product.
            uⱼ = u(αⱼ)

            # Record individual product utility.
            exp_uⱼ[j] = exp(uⱼ)

            # Add to total utility term.
            ∑exp_uⱼ += exp(uⱼ)
        end

        # Adjust utilities based on choice set.
        individual_choice_set_index = choice_sets[i,t]
        choice_set = all_choice_sets[individual_choice_set_index]
        exp_uⱼ = exp_uⱼ .* choice_set
        ∑exp_uⱼ = sum(exp_uⱼ)

        # Adjust for individuals with the empty choice set to avoid numerical issues.
        if individual_choice_set_index == 1
            ∑exp_uⱼ = 1.0
        end

        # Calculate the true choice probabilities.
        true_choice_frequencies = exp_uⱼ ./ ∑exp_uⱼ

        # Sample choices according to these frequencies.
        observed_choices = sample(1:n_products, Weights(true_choice_frequencies), Choice_Count[i,t])

        # Calculate the observed choice probabilities.
        observed_choice_probs = zeros(n_products)

        for choice in observed_choices
            for j in 1:n_products
                if choice == j
                    observed_choice_probs[j] += 1
                end
            end
        end

        # Adjust for individuals with the empty choice set to avoid numerical issues.
        ∑observed_choice_probs = sum(observed_choice_probs)
        if ∑observed_choice_probs == 0; ∑observed_choice_probs = 1.0 end

        observed_choice_probs = observed_choice_probs ./ ∑observed_choice_probs
        Choice_Frequency[i,t,:] = deepcopy(observed_choice_probs)
    end
end

# Package the parameters and data into structs.
@with_kw mutable struct Parameters_Struct
    utility_parameters::Array{Float64, 1}
    choice_set_parameters::Array{Float64, 1}
end

@with_kw mutable struct Parameters_Struct_Any
    utility_parameters::Array{Any, 1}
    choice_set_parameters::Array{Any, 1}
end

@with_kw struct Environment_Struct
    all_choice_sets::Array{Array{Int64}, 1}
    n_individuals::Int64
    n_periods::Int64
    n_products::Int64
end

@with_kw struct Data_Struct
    Choice_Count::Array{Int64, 2}
    Choice_Frequency::Array{Float64, 3}
    Choice_Set_Observable::Array{Float64, 3}
end

parameters  = Parameters_Struct(utility_parameters, choice_set_parameters)
environment = Environment_Struct(all_choice_sets, n_individuals, n_periods, n_products)
data        = Data_Struct(Choice_Count, Choice_Frequency, Choice_Set_Observable)


# -----------------------------------------------------------------------------
# 2.0 - Key Functions.
# -----------------------------------------------------------------------------
# Log likelihood for a single individual in a single period.
function log_likelihood(individual_index, period_index, parameters, data, environment)

    # Unpack relevant parameters.
    @unpack utility_parameters, choice_set_parameters = parameters 
    @unpack all_choice_sets, n_individuals, n_periods, n_products = environment
    @unpack Choice_Count, Choice_Frequency, Choice_Set_Observable = data

    # Get individual parameters.
    i = individual_index
    t = period_index

    # Get the type of variable values: either Float64 (most cases) or ForwardDiff.dual (during derivative calculations).
    value_type = eltype(utility_parameters[1])

    # Track log likelihood.
    ℓ = zero(value_type) # equivalent to ℓ = 0.0

    # Calculate the utility for each choice.
    exp_uⱼ = Vector{value_type}(undef, n_products)
    for j in 1:n_products

        # Get utility parameters.
        if j == n_products
            αⱼ = 0.0 # normalize utility for third product
        else
            αⱼ = utility_parameters[j]
        end

        uⱼ = u(αⱼ)
        exp_uⱼ[j] = exp(uⱼ)
    end

    # Get the probability of each choice set.
    choice_set_probs = get_choice_set_probabilities(individual_index, period_index, parameters, data, environment)

    # Calculate the expected likelihood of each observed choice.
    for j in 1:n_products

        # Calculate the odds of choosing this product under each choice set.
        expected_likelihood = 0.0

        for choice_set_index in eachindex(all_choice_sets)

            choice_set = all_choice_sets[choice_set_index]

            # If the product is not in this choice set, move on to the next choice set.
            if choice_set[j] == 0
                continue
            end

            # Otherwise, calculate the likelihood of choosing product j.
            ∑exp_uⱼ = dot(exp_uⱼ, choice_set)
            likelihood = exp_uⱼ[j] / ∑exp_uⱼ # logit CCP likelihood

            # Add to expected likelihood.
            expected_likelihood += likelihood * choice_set_probs[choice_set_index]
        end

        # Correction for expected likelihood if it's numerically rounded to zero.
        if expected_likelihood == 0.0; expected_likelihood = eps(0.0) end # smallest float 

        # Get the number of observations where product j was chosen.
        choice_frequency = Choice_Frequency[i,t,j]
        total_choices = Choice_Count[i,t]
        j_choices = choice_frequency * total_choices

        # Add the expected likelihood to the total for all corresponding choices.
        ℓ += log(expected_likelihood) * j_choices
    end

    return ℓ
end

# Log likelihood for a single individual across all periods.
function log_likelihood_individual(individual_index, parameters, data, environment)

    i = individual_index
    ℓ = 0.0

    # Calculate log likelihood for all observations.
    for t in 1:environment.n_periods
        ℓ += log_likelihood(i, t, parameters, data, environment)
    end

    return ℓ
end

# Log likelihood for all individuals across a single period.
function log_likelihood_period(period_index, parameters, data, environment)

    t = period_index
    ℓ = 0.0

    # Calculate log likelihood for all observations.
    for i in 1:environment.n_individuals
        ℓ += log_likelihood(i, t, parameters, data, environment)
    end

    return ℓ
end

# Log likelihood for all observations.
function log_likelihood_all(parameters, data, environment)

    ℓ = 0.0

    # Calculate log likelihood for all observations.
    for i in 1:environment.n_individuals
        for t in 1:environment.n_periods
            ℓ += log_likelihood(i, t, parameters, data, environment)
        end
    end

    return ℓ
end

# Calculate the choice set probabilities for a given individual.
function get_choice_set_probabilities(individual_index, period_index, parameters, data, environment)

    # Unpack relevant parameters.
    @unpack choice_set_parameters = parameters
    @unpack Choice_Set_Observable = data
    @unpack all_choice_sets, n_products = environment
    
    # Individual parameters.
    i = individual_index
    t = period_index

    # Get the type of variable values: either Float64 (most cases) or ForwardDiff.dual (during derivative calculations).
    value_type = eltype(choice_set_parameters[1])

    # Calculate the probability of each product being in a choice set.
    choice_probs = Vector{value_type}(undef, n_products)
    for j in 1:n_products
        choice_probs[j] = ϕ(Choice_Set_Observable[i,t,j], choice_set_parameters)
    end

    # Calculate the likelihood of each choice set.
    choice_set_probs = Vector{value_type}(undef, size(all_choice_sets))

    for choice_set_index in eachindex(all_choice_sets)
        
        choice_set = all_choice_sets[choice_set_index]

        # Calculate the likelihood of this choice set.
        choice_set_probability = 1.0
        for j in eachindex(choice_probs)

            # Check if the product is in the choice set.
            if choice_set[j] == 1
                choice_set_probability *= choice_probs[j]
            else
                choice_set_probability *= 1 - choice_probs[j]
            end
        end
        choice_set_probs[choice_set_index] = choice_set_probability
    end

    return choice_set_probs
end

# Prior distribution for a single parameter.
function log_prior(parameter_field, parameter_index, parameters)

    # Get the relevant parameter value.
    parameter_value = getfield(parameters, parameter_field)[parameter_index]
   
    # Calculate log likelihood.
    ℓ = log(pdf(Normal(0, 20^2), parameter_value)) # diffuse normal prior (input is SD, not variance)

    return ℓ
end


# -----------------------------------------------------------------------------
# Helper functions used throughout.

# Copying custom structs.
function copy_struct(parameters::Parameters_Struct)

    @unpack utility_parameters, choice_set_parameters = parameters

    return Parameters_Struct(deepcopy(utility_parameters),
            deepcopy(choice_set_parameters))
end

function copy_struct(parameters::Parameters_Struct_Any)

    @unpack utility_parameters, choice_set_parameters = parameters

    return Parameters_Struct_Any(deepcopy(utility_parameters), 
                deepcopy(choice_set_parameters))
end

# Fill a struct's fields all with the same value.
function fill_struct!(Struct, fill_value)
    for field in fieldnames(typeof(Struct))

        field_value = getfield(Struct, field)

        # Fill in arrays.
        if field_value isa AbstractArray

            # Skip this field if it contains values of different types; avoid filling Int field with Floats, for example.
            if typeof(field_value[1]) != typeof(fill_value)
                continue
            else
                field_value .= fill_value
            end

        # Fill in scalar values.
        else

            # Skip this field if it contains values of different types; avoid filling Int field with Floats, for example.
            if typeof(field_value) != typeof(fill_value)
                continue
            else
                setfield!(Struct, field, fill_value)
            end
        end
    end
end

# Convert a parameters struct into a vector of all the values containted in the struct.
function struct_to_vector(parameters)

    struct_values = [vec(getfield(parameters, field)) for field in fieldnames(typeof(parameters))]
    values_vector = vcat(struct_values...)

    return values_vector
end

# Convert from a vector of values to a parameters struct.
function vector_to_struct!(vector, output_struct)

    next_vector_index = 1

    # Fill in each field of the output struct.
    for field in fieldnames(typeof(output_struct))

        # Get the size of the given field.
        field_values = getfield(output_struct, field)
        field_length = size(field_values[:], 1)

        # Get the relevant values from the vector and reshape to fit the field.
        vector_values = vector[next_vector_index : next_vector_index + field_length - 1]

        # Fill in values from the vector.
        for value_index in eachindex(field_values)
            field_values[value_index] = vector_values[value_index]
        end

        # Update the position of used data in the vector.
        next_vector_index = next_vector_index + field_length
    end
end


# -----------------------------------------------------------------------------
# Functions used in maximum likelihood estimation.
parameters_any_sample = Parameters_Struct_Any(utility_parameters, choice_set_parameters)

# Likelihood function that takes a vector of parameter values as input.
function likelihood_objective(parameter_vector, sample_parameter_struct, data, environment)

    # Convert parameter vector into a struct.
    temp_parameter_struct = copy_struct(sample_parameter_struct)
    vector_to_struct!(parameter_vector, temp_parameter_struct)

    # Get the likelihood of this set of parameters.
    ℓ = log_likelihood_all(temp_parameter_struct, data, environment)

    return ℓ
end

# Wrapper function with only the parameter vector as input.
function likelihood_objective(parameter_vector)
    return likelihood_objective(parameter_vector, parameters_any_sample, data, environment)
end

# Numerical likelihood gradient function.
function likelihood_gradient(parameter_vector)
   return ForwardDiff.gradient(likelihood_objective, parameter_vector)
end

# Numerical likelihood Hessian function.
function likelihood_hessian(parameter_vector)
    return ForwardDiff.hessian(likelihood_objective, parameter_vector)
end

# -----------------------------------------------------------------------------
# 3.0 - Test likelihood function.
# -----------------------------------------------------------------------------
# Plot the likelihood function around the true value for each parameter.
for field in fieldnames(Parameters_Struct)

    parameter_values_array = getfield(parameters, field)
    
    for parameter_index in eachindex(parameter_values_array)

        # Get the true parameters.
        plotting_parameters = copy_struct(parameters)

        true_value = parameter_values_array[parameter_index]
        value_range = LinRange(true_value * 0.25, true_value * 4.0, 50) # values around the true value
        #value_range = LinRange(-10.0, 10.0, 50) # values around the true value
        likelihood_range = similar(value_range)

        # Calculate the likelihood for each value in the parameter range.
        for value_index in eachindex(value_range)
            parameter_value = value_range[value_index]
            getfield(plotting_parameters, field)[parameter_index] = parameter_value
            likelihood = log_likelihood_all(plotting_parameters, data, environment)
            likelihood_range[value_index] = likelihood
        end

        # Plot the likelihood for each value in the parameter range.
        fig = plot(value_range, likelihood_range, label="", title="$field: index: $parameter_index")
        plot!([true_value, true_value], [minimum(likelihood_range), maximum(likelihood_range)],
        label="true value", line=:dash, color=:black)
        display(fig)
    end
end


# -----------------------------------------------------------------------------
# 4.0 - Maximum likelihood estimation.
# -----------------------------------------------------------------------------
# Guess parameter values.
utility_parameters_guess = utility_parameters * 1.1
choice_set_intercepts_guess = choice_set_parameters * 0.9
parameters_guess = Parameters_Struct_Any(utility_parameters_guess, choice_set_intercepts_guess)

parameters_guess = copy_struct(parameters)
vector_guess = struct_to_vector(parameters_guess)

θₙ = vector_guess
θₙ₊₁ = similar(θₙ)

# Search parameters.
distance = Inf
distance_target = 1e-5
iteration_count = 0
max_iterations = 200
stop_condition_met = false
converged = false
step_size = 0.1

# Main loop.
while(stop_condition_met == false)

    # Get next set of parameter values.
    ∇ = likelihood_gradient(θₙ)
    H = likelihood_hessian(θₙ)
    θₙ₊₁ = θₙ - step_size * inv(H) * ∇

    # Update search parameters.
    distance = maximum(abs.(θₙ₊₁ - θₙ))
    θₙ = deepcopy(θₙ₊₁)
    iteration_count += 1

    # Check stopping conditions.
    if distance < distance_target
        stop_condition_met = true
        converged = true
    elseif iteration_count >= max_iterations
        stop_condition_met = true
        converged = false
    end

    # Report progress.
    println("Iteration $iteration_count, Distance = $distance, θ = $(θₙ)")
    if converged == true
        println("Converged after $iteration_count iterations to distance $distance")
    elseif (stop_condition_met == true) & (converged == false)
        println("Failed to converge after $iteration_count iterations. Distance = $distance")
    end
end

# Estimate standard errors.
H = likelihood_hessian(θₙ)
SE = sqrt.(diag(inv(-H)))

# Parameter labels.
parameter_labels = []
for field in fieldnames(Parameters_Struct)
    values = getfield(parameters, field)
    for index in eachindex(values)
        push!(parameter_labels, "$field index $index")
    end
end
parameter_labels = parameter_labels

# Report results.
results = [θₙ θₙ .- 1.96*SE θₙ .+ 1.96*SE]
vector_true_values = struct_to_vector(parameters)
println("\n Maximum Likelihood Parameter Estimates")
display(DataFrame([parameter_labels vector_guess vector_true_values results],
                  ["variable", "guess", "true value", "estimated mean", 
                  "95% Confidence Interval Low", "95% Confidence Interval High"]))