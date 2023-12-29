#=
Built off of 7.3.

Key features
- Detailing payments affect choice set probability.
- Choice sets can change across periods.
- Mixture model where types are defined by their choice sets.
- Quality parameters are shared across all types.
- Logit model with heterogeneous individuals. 
- Simulated data.
- Random walk variance adjusted for each individual. 

Individuals are randomly assigned a choice set each period, which may be a subset of all
available choices. "Choice set membership" is a new random variable which is 
estimated with Bayesian MCMC. It is a discrete distribution across the indices 
of choice sets. The probability is influenced by the amount of payments in that period.

The simulated data here is of the same form as the real data.
The real data has aggregated drug choice counts and payment values for each year.
The simulated data mimics this form. It draws total prescription counts and 
calculates drug choice frequencies.

Version 7.31 (with real data) has some minor performance optimizations.

Drug quality parameter for each drug is heterogeneous across physicians.

Observe individual product choices (not choice probabilities).

The model is estimated with Bayesian MCMC and numerical likelihood maximization.

Payments are roughly exponentially distributed from 10-1000.
Total claims are roughly exponentially distributed from 1-200.
Price is calculated as Total_30_day_fills / Total_Cost.
=#

using Distributions, Parameters, StatsBase, DataFrames, Plots, BenchmarkTools,
    LinearAlgebra, Statistics, ForwardDiff

# -----------------------------------------------------------------------------
# 1.0 - Model Setup and Simulate Data.
# -----------------------------------------------------------------------------
# Environment.
n_individuals = 2000
n_periods = 6
n_products = 3

# Utility parameters.
utility_parameters = [0.5, 1.0]  # third choice has normalized utility of 0.0
u(αᵢⱼ) = αᵢⱼ   # utility function

# Choice sets.
all_choice_sets =  [[0,0,0],
                    [0,0,1], [0,1,0], [1,0,0],
                    [0,1,1], [1,0,1], [1,1,0],
                    [1,1,1]]

choice_set_intercepts = Array{Float64, 3}(undef, (n_periods, n_products, 2))
choice_set_intercepts = [[0.5 1.0 0.5];
                         [0.5 1.0 0.5];
                         [1.5 1.5 0.5];
                         [1.5 1.5 0.5];
                         [2.0 1.0 3.0];
                         [2.0 1.0 3.0]]  ./ 5.0 # intercept/product-specific term

choice_set_payments = [0.10] # impact of payments on choice set probability
#choice_set_intercepts[:,1] .= 1.0
#choice_set_intercepts[:,2] .= 2.0
#choice_set_intercepts[:,3] .= 3.0


# Observable: payments.
Payments = log.(rand(Exponential(1000), (n_individuals, n_periods, n_products))) .*
(rand(Uniform(0,1), (n_individuals, n_periods, n_products)) .> 0.40)

# Draw true individual choice sets.
choice_sets = Array{Int64, 2}(undef, n_individuals, n_periods) # store the index of choice set for each i, t. 

for i in 1:n_individuals
    for t in 1:n_periods

        choice_set_probs = Vector{Float64}(undef, size(all_choice_sets,1))

        # Calculate the choice set probabilities adjusted by total payments. See Goeree 2008.
        for choice_set_index in eachindex(all_choice_sets)
            
            choice_set = all_choice_sets[choice_set_index]
            drug_probs = zeros(size(choice_set))

            for j in 1:n_products
               
                # Get the product and detailing coefficients.
                γₜⱼ = choice_set_intercepts[t,j]
                λᵢₜⱼ = Payments[i,t,j] * choice_set_payments[1]

                # Calculate probability of each drug being in the choice set.
                drug_probs[j] = exp(γₜⱼ + λᵢₜⱼ) / (1.0 + exp(γₜⱼ + λᵢₜⱼ))
            end

            # Calculate the likelihood of this choice set.
            choice_set_probability = 1.0
            for j in eachindex(drug_probs)

                # Check if the product is in the choice set.
                if choice_set[j] == 1
                    choice_set_probability *= drug_probs[j]
                else
                    choice_set_probability *= 1 - drug_probs[j]
                end
            end
            choice_set_probs[choice_set_index] = choice_set_probability
        end
    
        # Select choice set according to probabilities.
        choice_sets[i,t] = sample(eachindex(all_choice_sets), Weights(choice_set_probs))
    end
end

# Observable: prescription count and choice frequencies.
Prescription_Count = Array{Int64, 2}(undef, n_individuals, n_periods)
Prescription_Choice_Frequency = Array{Float64, 3}(undef, n_individuals, n_periods, n_products)

# Draw prescription counts.
for i in 1:n_individuals
    for t in 1:n_periods
        if choice_sets[i,t] == 1            # empty choice set
            Prescription_Count[i,t] = 0
        else                                # non-empty choice set
            Prescription_Count[i,t] = rand(Geometric(1/200))
        end
    end
end

# Calculate drug choice frequencies.
for i in 1:n_individuals
    for t in 1:n_periods

        # Terms in the logit CCP expression.
        exp_uⱼ = zeros(n_products)
        ∑exp_uⱼ = 0.0

        # Calculate utility for each drug choice.
        for j in 1:n_products

            # Get parameters and payments.
            if j == n_products
                αⱼ = 0.0 # normalize utility for third product
            else
                αⱼ = utility_parameters[j]
            end
            payment = Payments[i,t,j]

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
        observed_choices = sample(1:n_products, Weights(true_choice_frequencies), Prescription_Count[i,t])

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
        Prescription_Choice_Frequency[i,t,:] = deepcopy(observed_choice_probs)
    end
end

# Package the parameters and data into structs.
@with_kw mutable struct Parameters_Struct
    utility_parameters::Array{Float64, 1}
    choice_set_intercepts::Array{Float64, 2}
    choice_set_payments::Array{Float64, 1}
end

@with_kw mutable struct Parameters_Struct_Any
    utility_parameters::Array{Any, 1}
    choice_set_intercepts::Array{Any, 2}
    choice_set_payments::Array{Any, 1}
end

@with_kw struct Environment_Struct
    all_choice_sets::Array{Array{Int64}, 1}
    n_individuals::Int64
    n_periods::Int64
    n_products::Int64
end

@with_kw struct Data_Struct
    Payments::Array{Float64, 3}
    Prescription_Count::Array{Int64, 2}
    Prescription_Choice_Frequency::Array{Float64, 3}
end

parameters  = Parameters_Struct(utility_parameters, choice_set_intercepts, choice_set_payments)
environment = Environment_Struct(all_choice_sets, n_individuals, n_periods, n_products)
data        = Data_Struct(Payments, Prescription_Count, Prescription_Choice_Frequency)


# -----------------------------------------------------------------------------
# 2.0 - Key Functions.
# -----------------------------------------------------------------------------
# Log likelihood for a single individual in a single period.
function log_likelihood(individual_index, period_index, parameters, data, environment)

    # Unpack relevant parameters.
    @unpack utility_parameters, choice_set_intercepts, choice_set_payments = parameters 
    @unpack all_choice_sets, n_individuals, n_periods, n_products = environment
    @unpack Payments, Prescription_Count, Prescription_Choice_Frequency = data

    # Get individual parameters.
    i = individual_index
    t = period_index

    # Get the type of variable values: either Float64 (most cases) or ForwardDiff.dual (during derivative calculations).
    value_type = eltype(choice_set_intercepts[1])

    # Track log likelihood.
    ℓ = zero(value_type) # equivalent to ℓ = 0.0

    # Calculate the utility for each choice.
    exp_uⱼ = Vector{value_type}(undef, n_products)
    for j in 1:n_products

        # Get parameters and payments.
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

        # Get the number of observations where drug j was chosen.
        choice_frequency = Prescription_Choice_Frequency[i,t,j]
        total_choices = Prescription_Count[i,t]
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
    @unpack choice_set_intercepts, choice_set_payments = parameters
    @unpack Payments = data
    @unpack all_choice_sets, n_products = environment
    
    # Individual parameters.
    i = individual_index
    t = period_index

    # Get the type of variable values: either Float64 (most cases) or ForwardDiff.dual (during derivative calculations).
    value_type = eltype(choice_set_intercepts[1])

    # Calculate the probability of each product being in a choice set.
    drug_probs = Vector{value_type}(undef, n_products)

    for j in 1:n_products
        
        # Get the product and detailing coefficients.
        γₜⱼ = choice_set_intercepts[t,j]
        λᵢₜⱼ = Payments[i,t,j] * choice_set_payments[1]

        # Calculate probability of each drug being in the choice set.
        drug_probs[j] = exp(γₜⱼ + λᵢₜⱼ) / (1.0 + exp(γₜⱼ + λᵢₜⱼ))
    end

    # Calculate the likelihood of each choice set.
    choice_set_probs = Vector{value_type}(undef, size(all_choice_sets))

    for choice_set_index in eachindex(all_choice_sets)
        
        choice_set = all_choice_sets[choice_set_index]

        # Calculate the likelihood of this choice set.
        choice_set_probability = 1.0
        for j in eachindex(drug_probs)

            # Check if the product is in the choice set.
            if choice_set[j] == 1
                choice_set_probability *= drug_probs[j]
            else
                choice_set_probability *= 1 - drug_probs[j]
            end
        end
        choice_set_probs[choice_set_index] = choice_set_probability
    end

    return choice_set_probs
end

#= Prior distribution for population & homogeneous parameters.
function log_prior(parameters)

    # Unpack relevant parameters.
    @unpack utility_parameters, choice_set_parameters = parameters
    n_qualtity_params = size(utility_parameters, 1)
    n_choice_set_params = size(choice_set_parameters[:], 1)
   
    # Calculate log likelihood using population distribution pdf. 
    ℓ = 0.0
    ℓ += log(pdf(MvNormal(zeros(n_qualtity_params),
            diagm(20*ones(n_qualtity_params))), utility_parameters))      # diffuse normal prior for quality
    ℓ += log(pdf(MvNormal(zeros(n_choice_set_params),
            diagm(20*ones(n_choice_set_params))), choice_set_parameters)) # diffuse normal prior for all choice set parameters

    return ℓ
end =#

# Prior distribution for a single parameter.
function log_prior(parameter_field, parameter_index, parameters)

    # Get the relevant parameter value.
    parameter_value = getfield(parameters, parameter_field)[parameter_index]
   
    # Calculate log likelihood.
    ℓ = log(pdf(Normal(0, 20^2), parameter_value)) # diffuse normal prior (input is SD, not variance)

    return ℓ
end

#= Get proposal for individual parameters.
function get_proposal!(parameters, ρ)

    # Unpack relevant parameters.
    @unpack α = parameters

    # MH random walk for α parameters.
    α_proposed = similar(α)
    for α_index in eachindex(α_proposed)
        α_proposed[α_index] = α[α_index] + rand(Normal(0,ρ))
    end
    α_proposed[lastindex(α_proposed)] = 0.0 # Correct for unidentified coefficient.

    # Update parameter struct with proposed parameters.
    parameters.α = copy(α_proposed)
end
=#

# -----------------------------------------------------------------------------
# Helper functions used throughout.

# Copying custom structs.
function copy_struct(parameters::Parameters_Struct)

    @unpack utility_parameters, choice_set_intercepts, choice_set_payments = parameters

    return Parameters_Struct(deepcopy(utility_parameters),
                             deepcopy(choice_set_intercepts),
                             deepcopy(choice_set_payments))
end

function copy_struct(parameters::Parameters_Struct_Any)

    @unpack utility_parameters, choice_set_intercepts, choice_set_payments = parameters

    return Parameters_Struct_Any(deepcopy(utility_parameters),
                                 deepcopy(choice_set_intercepts),
                                 deepcopy(choice_set_payments))
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
parameters_any = Parameters_Struct_Any(utility_parameters, choice_set_intercepts, choice_set_payments)

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
    return likelihood_objective(parameter_vector, parameters_any, data, environment)
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

#=
# -----------------------------------------------------------------------------
# 4.0 - Maximum likelihood estimation.
# -----------------------------------------------------------------------------
# Guess parameter values.
utility_parameters_guess = utility_parameters * 1.1
choice_set_intercepts_guess = choice_set_intercepts * 0.9
choice_set_payments_guess = choice_set_payments * 0.9
parameters_guess = Parameters_Struct(utility_parameters_guess,
                                    choice_set_intercepts_guess,
                                    choice_set_payments_guess)

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
    println("Iteration $iteration_count, Distance = $distance, θ = $(θₙ[1:5])")
    if converged == true
        println("Converged after $iteration_count iterations to distance $distance")
    elseif (stop_condition_met == true) & (converged == false)
        println("Failed to converge after $iteration_count iterations. Distance = $distance")
    end
end


# Debugging: estimate using Optim.
using Optim

optim_objective(x) = -likelihood_objective([vector_guess[1:9]; x; vector_guess[20:21]])
opt_options = Optim.Options(iterations=1000)
result = optimize(optim_objective, vector_guess[10:19], BFGS(), opt_options)

result.minimizer
result.minimum
=#


# -----------------------------------------------------------------------------
# 5.0 - Estimation with Bayesian MCMC. 
# -----------------------------------------------------------------------------
# MCMC parameters.
n_draws = 20000                                     # total draws
burn_in = 10000                                     # number of initial draws to ignore
ρ_start = 0.01                                    # MH random walk variance starting value for each parameter
acceptance_rate_low = 0.21                         # target acceptance rate for indiviudal parameters
acceptance_rate_high = 0.25

proposal_counts = zeros(n_draws)                    # store proposal counts for individual parameters
acceptance_rates = 0                                # store acceptance rates for individual parameters

# Use Model_Parameter_Structs to store random walk variance and acceptance rates for each parameter/individual/period.
ρ = copy_struct(parameters)
fill_struct!(ρ, ρ_start)
acceptance_rates = copy_struct(parameters)
fill_struct!(acceptance_rates, 1.0)

# Vectors to store draws.
parameter_draws = Vector{Parameters_Struct}(undef, n_draws)

# Parameter guess values.
#=parameter_guess = Parameters_Struct(zeros(n_individuals, n_products), # individual α
                                    zeros(n_individuals),             # individual β
                                    zeros(n_products), Matrix(I(3)),  # population α
                                    0.0, 1.0)                         # population β
=#
parameter_guess = copy_struct(parameters)           # true value, use for debugging
parameter_draws[1] = copy_struct(parameter_guess)

# Main loop.
for n in 2:n_draws

    # Get previous parameter draw. 
    parameter_draw_next = copy_struct(parameter_draws[n-1])

    # 1. Gibbs sampling. Nothing to sample in this model.


    # 2. Metropolis-Hastings sampling for all other parameters.
    
    # Classify parameters according to whether they vary across products, periods, individuals, etc. 
    use_all_likelihood = [:utility_parameters, :choice_set_payments]
    use_individual_likelihood = []
    use_period_likelihood = [:choice_set_intercepts]
    # use_individual_period_likelihood = []
    all_parameters = [use_all_likelihood; use_individual_likelihood; use_period_likelihood]
    #all_parameters = [:utility_parameters]

    # Begin sampling for all parameters.
    for parameter_field in all_parameters
        
        parameter_values = getfield(parameter_draw_next, parameter_field)

        for parameter_index in eachindex(parameter_values)

            # For heterogeneous parameters, get individual and period indices, if appropriate.
            individual_index = 0
            period_index = 0
            choice_index = 0

            indices = CartesianIndices(parameter_values)[parameter_index] # convert from 1 dimensional index to typical row/column indices

            if parameter_field ∈ use_period_likelihood
                period_index = indices[1]
            end

            # Likelihood and prior for previous set of parameters.
            log_likelihood_previous = 0.0
            if parameter_field ∈ use_all_likelihood
                log_likelihood_previous = log_likelihood_all(parameter_draw_next, data, environment)  
            elseif parameter_field ∈ use_period_likelihood
                log_likelihood_previous = log_likelihood_period(period_index, parameter_draw_next, data, environment)
            end

            log_prior_previous = log_prior(parameter_field, parameter_index, parameter_draw_next)

            # Save previous parameter value.
            previous_parameter_value = deepcopy(getfield(parameter_draw_next, parameter_field)[parameter_index])

            # Search for next parameter draw.
            accepted_draw = false
            proposal_count = 0

            while(accepted_draw == false)

                # Update proposal count and check limits.
                proposal_count += 1
                if (proposal_count > 100000) && (n <= burn_in)
                    println("warning: too many proposals")
                    break
                end

                # Get proposal and update parameter struct.
                ρᵢ = getfield(ρ, parameter_field)[parameter_index]
                proposed_value = previous_parameter_value + rand(Normal(0, ρᵢ))                
                getfield(parameter_draw_next, parameter_field)[parameter_index] = proposed_value

                # Calculate likelihoods with the proposed parameter draw.
                log_likelihood_proposed = 0.0
                if parameter_field ∈ use_all_likelihood
                    log_likelihood_proposed = log_likelihood_all(parameter_draw_next, data, environment)
                elseif parameter_field ∈ use_period_likelihood
                    log_likelihood_proposed = log_likelihood_period(period_index, parameter_draw_next, data, environment)
                end

                log_prior_proposed = log_prior(parameter_field, parameter_index, parameter_draw_next)

                # Acceptance ratio.
                r = exp((log_likelihood_proposed + log_prior_proposed) -
                            (log_likelihood_previous + log_prior_previous))

                # Decide whether to accept.
                if (r > 1.0) | (r > rand(Uniform(0,1)))
                    accepted_draw = true
                else
                    # Reset parameter to previous value.
                    getfield(parameter_draw_next, parameter_field)[parameter_index] = previous_parameter_value
                end

                # Update random walk variance to meet target acceptance rate.
                if accepted_draw == true

                    # Calculate the acceptance rate within a given draw range.
                    draw_range = 100 # amount of previous draws to use in acceptance rate calculation
                    
                    accepted_proposal_count = min(draw_range, n)
                    previous_acceptance_rate = getfield(acceptance_rates, parameter_field)[parameter_index]
                    updated_acceptance_rate = (accepted_proposal_count + 1) /
                                            ((accepted_proposal_count/previous_acceptance_rate) + proposal_count)

                    # Increase or decrease the MH random walk variance accordingly.
                    if updated_acceptance_rate < acceptance_rate_low
                        getfield(ρ, parameter_field)[parameter_index] = deepcopy(ρᵢ) * 0.999
                    end
                    if updated_acceptance_rate > acceptance_rate_high
                        getfield(ρ, parameter_field)[parameter_index] = deepcopy(ρᵢ) * 1.001
                    end
                    
                    # Record updated acceptance rate.
                    getfield(acceptance_rates, parameter_field)[parameter_index] = updated_acceptance_rate
                end
            end
        end
    end     

    # Record the accepted full parameter draw.
    parameter_draws[n] = copy_struct(parameter_draw_next)

    # Display progress.
    if mod(n, 100) == 0
        println("completed $n of $n_draws draws, acceptance rates: $acceptance_rates")
    end
end

# -----------------------------------------------------------------------------
# 4.0 - Analyze and Display MCMC Results.
# -----------------------------------------------------------------------------
# Drop burn-in draws from the vector of accepted draws.
parameter_draws = parameter_draws[burn_in + 1 : n_draws]
n_draws = n_draws - burn_in

# True population parameter values.
true_parameters = struct_to_vector(parameters)
parameter_labels = []
for field in fieldnames(Parameters_Struct)
    values = getfield(parameters, field)
    for index in eachindex(values)
        push!(parameter_labels, "$field index $index")
    end
end

# Parameter draws.
population_parameter_draws = zeros(n_draws, size(true_parameters, 1))
for draw_index in 1:n_draws
    draws_vector = struct_to_vector(parameter_draws[draw_index])
    population_parameter_draws[draw_index, :] = draws_vector
end

# Table of parameter estimates including confidence sets. 
mean_observed = mean(population_parameter_draws, dims=1)
variance_observed = var(population_parameter_draws, dims=1)
stdev_observed = std(population_parameter_draws, dims=1)
bottom_95_CI = sort(population_parameter_draws, dims=1)[Int(round(0.025*size(population_parameter_draws, 1))), :]
top_95_CI = sort(population_parameter_draws, dims=1)[Int(round(0.975*size(population_parameter_draws, 1))), :]

results = [true_parameters mean_observed' bottom_95_CI top_95_CI]
println("\n Parameter Estimates")
display(DataFrame([parameter_labels results],
                  ["variable", "true value", "estimated mean", 
                  "95% Credible Interval Low", "95% Credible Interval High"]))

# Plots of accepted draws for population parameters.
for parameter_index in eachindex(parameter_labels)
    figure = plot(1:(n_draws), population_parameter_draws[:,parameter_index],
    label = "accepted draws", legend=:right)
    plot!([0, n_draws],[true_parameters[parameter_index], true_parameters[parameter_index]],
    label = "true parameter value")
    title!("$(parameter_labels[parameter_index]) accepted MCMC draws")
    plot!([0],[0], label="")
    display(figure)
end

# Show change in choice set probabilities over periods.
parameter_estimates = copy_struct(parameters)
vector_to_struct!(mean_observed, parameter_estimates)
all_choice_set_probs = zeros(size(all_choice_sets, 1), n_periods)
for t in 1:n_periods
    average_choice_set_probs = zeros(size(all_choice_sets, 1))
    for i in 1:n_individuals
        choice_set_probs = get_choice_set_probabilities(i, t, parameter_estimates, data, environment)
        average_choice_set_probs .+= choice_set_probs / n_individuals
    end
    all_choice_set_probs[:,t] = average_choice_set_probs
end

fig = plot(all_choice_set_probs', labels="",
    xlabel = "Period", ylabel = "Frequency", title = "Change in Choice Set Frequency Over Time",
    linestyle=[:solid :solid :solid :dash :dash :dash :dot], 
    linewidth = 2,
    linecolor = [:blue :red :black :blue :red :black :black],
    linealpha = 0.6)
plot!([n_periods + 1], [maximum(all_choice_set_probs)], label="", linewidth = 0) # expand plot
for choice_set_index in eachindex(all_choice_sets) # annotate 
    annotate!(n_periods + 0.5, all_choice_set_probs[choice_set_index, n_periods],
    text(string(all_choice_sets[choice_set_index]), 8, :black))
end
display(fig)



# Plot the maximum likelihood functions for debugging.    
true_values = struct_to_vector(parameters)

for parameter_index in eachindex(true_values)
    true_value = true_values[parameter_index]
    x_range = LinRange(true_value * 0.25, true_value * 4.0, 50)
    y = similar(x_range)

    for index in eachindex(x_range)
        parameter_vector = struct_to_vector(parameters)
        parameter_vector[parameter_index] = x_range[index]
        y[index] = likelihood_objective(parameter_vector, parameters, data, environment)
        #y[index] = likelihood_gradient(parameter_vector)[parameter_index]
    end
    display(plot(x_range, y, title="parameter $parameter_index, true value $true_value"))
end
