#=
Logit with heterogeneous individuals.
See documentation.
=#

using CSV, Distributions, Parameters, DataFrames, Plots, BenchmarkTools, LinearAlgebra

# -----------------------------------------------------------------------------
# 1.0 - Model Setup and Simulate Data.
# -----------------------------------------------------------------------------
# Environment.
n_individuals = 100
n_periods = 6
n_products = 3

# Quality parameters.
α_mean = [0.0, 1.0, 2.0]                                    
α_vcov = [1.0 0.0 0.0;
          0.0 1.0 0.0;
          0.0 0.0 1.0]
α = Matrix(rand(MvNormal(α_mean, α_vcov), n_individuals)')  # sample individual parameters
α[:,1] .= 0.0 # normalized choice / outside option.
n_quality_params = size(α_mean, 1)

# Payment parameters.
β_mean = -1.0
β_var  = 0.2
β = rand(Normal(β_mean, sqrt(β_var)), n_individuals)

# Utility function.
u(payment, αᵢⱼ, βᵢ) = αᵢⱼ + exp(βᵢ)*payment
# = (drug utility) + (payments)

# Draws for T1EV error.
ϵ = rand(Gumbel(0,1), (n_individuals, n_periods, n_products))

# Observables: payments.
Payments = log.(rand(Exponential(1000), (n_individuals, n_periods, n_products)))
# Optional: make payments correlated within an individual.
for t in 2:n_periods
    for i in 1:n_individuals
        for j in 1:n_products
            Payments[i,t,j] = Payments[i,t-1,j]*0.95 + log(rand(Exponential(1000)))*0.05
            # mixture of (previous value) and (random draw)
        end
    end
end

# Observables: prescription count and choice frequencies.
Prescription_Count = Array{Int64, 2}(undef, n_individuals, n_periods)
Prescription_Choice_Frequency = Array{Float64, 3}(undef, n_individuals, n_periods, n_products)

# Draw prescription counts.
for i in 1:n_individuals
    for t in 1:n_periods
        Prescription_Count[i,t] = rand(Geometric(1/200))
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
            αᵢⱼ = α[i,j]
            βᵢ = β[i]
            payment = Payments[i,t,j]

            # Get utility for this product.
            uⱼ = u(payment, αᵢⱼ, βᵢ)

            # Record individual product utility.
            exp_uⱼ[j] = exp(uⱼ)

            # Add to total utility term.
            ∑exp_uⱼ += exp(uⱼ)
        end

        # Calculate choice probabilities.
        Prescription_Choice_Frequency[i,t,:] = exp_uⱼ ./ ∑exp_uⱼ
    end
end

# Package the parameters and data into structs.
@with_kw mutable struct Parameters_Struct
    α::Matrix{Float64}
    β::Vector{Float64}
    α_mean::Vector{Float64}
    α_vcov::Matrix{Float64}
    β_mean::Float64
    β_var::Float64
end

@with_kw struct Environment_Struct
    n_individuals::Int64
    n_periods::Int64
    n_products::Int64
    exp_uⱼ::Vector{Float64} # preallocated vector used in likelihood calculation
end

@with_kw struct Data_Struct
    Payments::Array{Float64, 3}
    Prescription_Count::Array{Int64, 2}
    Prescription_Choice_Frequency::Array{Float64, 3}
end

parameters  = Parameters_Struct(α, β, α_mean, α_vcov, β_mean, β_var)
environment = Environment_Struct(n_individuals, n_periods, n_products, zeros(n_products))
data        = Data_Struct(Payments, Prescription_Count, Prescription_Choice_Frequency)


# -----------------------------------------------------------------------------
# 2.0 - Key Functions.
# -----------------------------------------------------------------------------
# Log likelihood for all observations.
function log_likelihood(parameters, data, environment)

    # Unpack relevant parameters.
    @unpack α, β = parameters 
    @unpack n_individuals, n_periods, n_products, exp_uⱼ = environment
    @unpack Payments, Prescription_Count, Prescription_Choice_Frequency = data

    ℓ = 0.0

    # Calculate log likelihood for all observations.
    for i in 1:n_individuals
        for t in 1:n_periods

            # Calculate utility for each drug choice.
            for j in 1:n_products

                # Get parameters and payments.
                αᵢⱼ = α[i,j]
                βᵢ = β[i]
                payment = Payments[i,t,j]

                # Get utility for this product.
                uⱼ = u(payment, αᵢⱼ, βᵢ)

                # Add to total utility term.
                exp_uⱼ[j] = exp(uⱼ)
            end

            # Get CCP denominator term.
            ∑exp_uⱼ = sum(exp_uⱼ)

            # Calculate likelihood contribution from each drug.
            for j in 1:n_products

                # Get the number of observations where drug j was chosen.
                choice_frequency = Prescription_Choice_Frequency[i,t,j]
                total_choices = Prescription_Count[i,t]
                j_choices = choice_frequency * total_choices

                # Calculate likelihood contribution (CCP * number of choices).
                ℓ += (log(exp_uⱼ[j]) - log(∑exp_uⱼ)) * j_choices   # Pr(j) = exp(μⱼ) / ∑ⱼexp(uⱼ)
            end
        end
    end

    return ℓ
end


# Log likelihood for a single individual
function log_likelihood(individual_index, parameters, data, environment)

    # Unpack relevant parameters.
    @unpack α, β = parameters 
    @unpack n_individuals, n_periods, n_products, exp_uⱼ = environment
    @unpack Payments, Prescription_Count, Prescription_Choice_Frequency = data

    i = individual_index
    ℓ = 0.0

    # Calculate log likelihood for all observations.
    for t in 1:n_periods

        # Calculate utility for each drug choice.
        for j in 1:n_products

            # Get parameters and payments.
            αᵢⱼ = α[i,j]
            βᵢ = β[i]
            payment = Payments[i,t,j]

            # Get utility for this product.
            uⱼ = u(payment, αᵢⱼ, βᵢ)

            # Add to total utility term.
            exp_uⱼ[j] = exp(uⱼ)
        end

        # Get CCP denominator term.
        ∑exp_uⱼ = sum(exp_uⱼ)

        # Calculate likelihood contribution from each drug.
        for j in 1:n_products

            # Get the number of observations where drug j was chosen.
            choice_frequency = Prescription_Choice_Frequency[i,t,j]
            total_choices = Prescription_Count[i,t]
            j_choices = choice_frequency * total_choices

            # Calculate likelihood contribution (CCP * number of choices).
            ℓ += (log(exp_uⱼ[j]) - log(∑exp_uⱼ)) * j_choices   # Pr(j) = exp(μⱼ) / ∑ⱼexp(uⱼ)
        end
    end

    return ℓ
end


# Prior distribution for population parameters.
function log_prior(parameters)

    # Unpack relevant parameters.
    @unpack α_mean, α_vcov, β_mean, β_var = parameters

    # Calculate log likelihood using diffuse prior for population parameters.
    ℓ = 0.0
    for parameter in [α_mean, α_vcov, β_mean, β_var]
        ℓ += log(pdf(Normal(0,10), parameter))
    end

    return ℓ
end

# Prior distribution for individual parameters.
function log_prior(individual_index, parameters)

    # Unpack relevant parameters.
    @unpack α, β, α_mean, α_vcov, β_mean, β_var = parameters
    
    # Parameters for this individual. 
    αᵢ = α[individual_index, :]
    βᵢ = β[individual_index]
    
    # Calculate log likelihood using population distribution pdf. 
    ℓ = 0.0
    ℓ += log(pdf(MvNormal(α_mean, α_vcov), αᵢ))
    ℓ += log(pdf(Normal(β_mean, sqrt(β_var)), βᵢ))

    return ℓ
end

# Get proposal for individual parameters.
function get_proposal!(individual_index, parameters, ρ)

    # Unpack relevant parameters.
    @unpack α, β = parameters

    # Parameters for this individual. 
    αᵢ = α[individual_index, :]
    βᵢ = β[individual_index]

    # MH random walk for α parameters.
    α_proposed = similar(αᵢ)
    for α_index in eachindex(α_proposed)
        α_proposed[α_index] = αᵢ[α_index] + rand(Normal(0,ρ[1]))
    end
    α_proposed[1] = 0.0 # Correct for unidentified coefficient.

    # Repeat for β parameter (with possibly different variance ρ).
    β_proposed = βᵢ + rand(Normal(0,ρ[2]))

    # Update parameter struct with proposed parameters.
    parameters.α[individual_index, :] = copy(α_proposed)
    parameters.β[individual_index] = copy(β_proposed)
end

# Generic function to allow for copying of custom structs.
function copy_struct(parameters::Parameters_Struct)

    @unpack α, β, α_mean, α_vcov, β_mean, β_var = parameters

    return Parameters_Struct(deepcopy(α), deepcopy(β),
                             deepcopy(α_mean), deepcopy(α_vcov),
                             deepcopy(β_mean), deepcopy(β_var))
end


# -----------------------------------------------------------------------------
# 3.0 - Perform Inference with Bayesian MCMC.
# -----------------------------------------------------------------------------
# MCMC parameters.
n_draws = 30000                                     # total draws
burn_in = 10000                                     # number of initial draws to ignore
ρ = repeat([0.05 0.05], n_individuals)              # MH random walk variance for individual parameters
acceptance_rate_low = 0.21                          # target range for acceptance rate (see Train 2009)
acceptance_rate_high = 0.25                         # target range for acceptance rate (see Train 2009)

proposal_counts = zeros(n_draws, n_individuals)     # store proposal counts
acceptance_rates = zeros(n_draws, n_individuals)    # store the acceptance rates

# Vectors to store draws.
parameter_draws = Vector{Parameters_Struct}(undef, n_draws)

# Parameter guess values.
parameter_guess = Parameters_Struct(zeros(n_individuals, n_quality_params), # individual α
                                    zeros(n_individuals),                   # individual β
                                    zeros(n_quality_params), 
                                        Matrix(I(n_quality_params)),        # population α
                                    0.0, 1.0)                               # population β
# parameter_guess = copy_struct(parameters) # start at true values (debugging)
parameter_draws[1] = copy_struct(parameter_guess)

# Main loop.
for n in 2:n_draws

    # Get previous parameter draw. 
    parameter_draw_next = copy_struct(parameter_draws[n-1])

    # 1. Sample population parameters (Gibbs sampling).
    # a. Quality mean (α_mean); known covariance matrix, unknown means. See Train 2009 page 295.
    ᾱ = vec(mean(parameter_draw_next.α, dims=1))            # sample mean of quality parameter
    ᾱ_var = parameter_draw_next.α_vcov ./ n_individuals     # variance of quality mean
    parameter_draw_next.α_mean = rand(MvNormal(ᾱ, ᾱ_var))

    # b. Qualtiy variance (α_vcov); known means, unknown covariance matrix. See Train 2009 page 297.
    v₁ = n_quality_params + n_individuals                         # degrees of freedom parameter
    S = cov(parameter_draw_next.α)                          # sample covariance matrix 
    s₁ = (n_products*I(n_quality_params) .+ n_individuals*S) / v₁ # scale parameter
    parameter_draw_next.α_vcov = rand(InverseWishart(v₁, s₁)) * n_individuals

    # c. Payments mean (β_mean); known variance, unknown mean. See Train 2009 page 295.
    b̄ = mean(parameter_draw_next.β)                         # sample mean of payment parameter
    b̄_var = parameter_draw_next.β_var / n_individuals       # variance of payment parameter mean
    parameter_draw_next.β_mean = rand(Normal(b̄, sqrt(b̄_var)))

    # d. Payments variance (β_var); known mean, unknown variance. See Train 2009 page 297.
    v₁ = 1 + n_individuals                         # degrees of freedom parameter
    S = var(parameter_draw_next.β)                 # sample variance
    s₁ = (1 .+ n_individuals*S) / v₁               # scale parameter
    parameter_draw_next.β_var = rand(InverseGamma(v₁, s₁)) * n_individuals

    # 2. Sample individual parameters with Metropolis-Hastings.
    for individual_index in 1:n_individuals
        
        # Get log likelihood and log prior for the previous draw.
        log_likelihood_previous = log_likelihood(individual_index, parameter_draw_next, data, environment)
        log_prior_previous = log_prior(individual_index, parameter_draw_next)

        # Search for next parameter draw.
        accepted_draw = false
        proposal_count = 0

        # Save previous individual parameters. 
        αᵢ_previous = parameter_draw_next.α[individual_index,:]
        βᵢ_previous = parameter_draw_next.β[individual_index]

        while (accepted_draw == false)

            # Update proposal count and check limits.
            proposal_count += 1
            if (proposal_count > 500) && (n <= burn_in)
                println("too many proposals")
                break
            end

            # Get proposal.
            ρᵢ = ρ[individual_index, :]
            get_proposal!(individual_index, parameter_draw_next, ρᵢ)

            # Calculate likelihoods.
            log_likelihood_proposed = log_likelihood(individual_index, parameter_draw_next, data, environment)
            log_prior_proposed = log_prior(individual_index, parameter_draw_next)

            # Acceptance ratio.
            r = exp(log_likelihood_proposed + log_prior_proposed 
                - log_likelihood_previous - log_prior_previous)

            # Decide whether to accept.
            if (r > 1.0) | (r > rand(Uniform(0,1)))
                accepted_draw = true
                proposal_counts[n, individual_index] = proposal_count
            else
                # Reset individual parameters to previous value. 
                parameter_draw_next.α[individual_index,:] = αᵢ_previous
                parameter_draw_next.β[individual_index] = βᵢ_previous
            end

            # Update individual random walk variance to meet target acceptance rate.
            if accepted_draw == true

                draw_range = 500 # amount of previous draws to use in acceptance rate calculation
                acceptance_rate = draw_range / 
                                sum(proposal_counts[max(2,n-draw_range):n, individual_index])
                
                # Increase or decrease the MH random walk variance accordingly.
                if acceptance_rate < acceptance_rate_low             
                    ρ[individual_index, :] = ρᵢ .* 0.999
                end
                if acceptance_rate > acceptance_rate_high
                    ρ[individual_index, :] = ρᵢ .* 1.001
                end
            end
        end
    end

    # Record the accepted full parameter draw.
    parameter_draws[n] = copy_struct(parameter_draw_next)

    # Display progress.
    if mod(n, 100) == 0

        # Acceptance rate moving average for a few individuals. 
        acceptance_rates = round.(1 ./ mean(proposal_counts[max(n-1000, 1):n, :], dims=1)[1, 1:10], digits=2)

        println("completed $n of $n_draws draws, acceptance rates: $acceptance_rates")
    end
end


# -----------------------------------------------------------------------------
# 4.0 - Analyze and Display Results.
# -----------------------------------------------------------------------------
# Drop results from the burn-in period.
parameter_draws = parameter_draws[burn_in + 1 : n_draws]
n_draws = size(parameter_draws, 1)

# True population parameter values.
true_population_parameters = [α_mean; diag(α_vcov); β_mean; β_var]
parameter_labels = ["α₁ mean", "α₂ mean", "α₃ mean", "α₁ var", "α₂ var", "α₃ var",
                    "β mean", "β var"]

# Population parameter draws.
population_parameter_draws = zeros(n_draws, size(true_population_parameters, 1))
for draw_index in 1:n_draws
    draw_parameters = parameter_draws[draw_index]

    population_parameter_draws[draw_index, :] = 
        [draw_parameters.α_mean; diag(draw_parameters.α_vcov);
         draw_parameters.β_mean; draw_parameters.β_var]
end

# Table of parameter estimates including confidence sets. 
mean_observed = mean(population_parameter_draws[burn_in:n_draws, :], dims=1)
variance_observed = var(population_parameter_draws[burn_in:n_draws, :], dims=1)
stdev_observed = std(population_parameter_draws[burn_in:n_draws, :], dims=1)
bottom_95_CI = sort(population_parameter_draws[burn_in:n_draws, :], dims=1)[Int(round(0.025*size(population_parameter_draws[burn_in:n_draws, :], 1))), :]
top_95_CI = sort(population_parameter_draws[burn_in:n_draws, :], dims=1)[Int(round(0.975*size(population_parameter_draws[burn_in:n_draws, :], 1))), :]

results = [true_population_parameters mean_observed' bottom_95_CI top_95_CI]
println("\n Parameter Estimates")
display(DataFrame([parameter_labels results],
                  ["variable", "true value", "estimated mean", 
                  "95% Credible Interval Low", "95% Credible Interval High"]))

# Plots of accepted draws for population parameters.
for parameter_index in eachindex(parameter_labels)
    figure = plot(1:(n_draws), population_parameter_draws[:,parameter_index],
    label = "accepted draws", legend=:right)
    plot!([0, n_draws],[true_population_parameters[parameter_index], true_population_parameters[parameter_index]],
    label = "true parameter value")
    title!("$(parameter_labels[parameter_index]) accepted MCMC draws")
    plot!([0],[0], label="")
    display(figure)
end

# Repeat for selected individual parameter values.
individual_indices = [1,5,10]
true_individual_parameters = reduce(vcat, [[parameters.α[i,:]; parameters.β[i]] for i in individual_indices])

parameter_labels = ["α₁", "α₂", "α₃", "β"]
parameter_labels_all = repeat(parameter_labels, size(individual_indices, 1))
individual_labels = repeat(1:size(individual_indices,1), inner=size(parameter_labels,1))

# Individual parameter draws.
individual_parameter_draws = zeros(n_draws, size(parameter_labels, 1), size(individual_indices,1))
for individual_index in eachindex(individual_indices)
    i = individual_indices[individual_index]
    for draw_index in 1:n_draws
        individual_parameter_draws[draw_index, :, individual_index] = 
        [parameter_draws[draw_index].α[i,:]; parameter_draws[draw_index].β[i]]
    end
end

# Combine along individual dimension.
individual_parameter_draws = reshape(individual_parameter_draws, (n_draws, size(parameter_labels_all, 1))) 

# Table of parameter estimates including confidence sets. 
mean_observed = mean(individual_parameter_draws[burn_in:n_draws, :], dims=1)
variance_observed = var(individual_parameter_draws[burn_in:n_draws, :], dims=1)
stdev_observed = std(individual_parameter_draws[burn_in:n_draws, :], dims=1)
bottom_95_CI = sort(individual_parameter_draws[burn_in:n_draws, :], dims=1)[Int(round(0.025*size(individual_parameter_draws[burn_in:n_draws, :], 1))), :]
top_95_CI = sort(individual_parameter_draws[burn_in:n_draws, :], dims=1)[Int(round(0.975*size(individual_parameter_draws[burn_in:n_draws, :], 1))), :]

results = [true_individual_parameters mean_observed' bottom_95_CI top_95_CI]
println("\n Parameter Estimates")
display(DataFrame([individual_labels parameter_labels_all results],
        ["individual", "variable", "true value", "estimated mean", 
        "95% Credible Interval Low", "95% Credible Interval High"]))
display(groupby(DataFrame([individual_labels parameter_labels_all results],
        ["individual", "variable", "true value", "estimated mean", 
        "95% Credible Interval Low", "95% Credible Interval High"]), :individual))


# Plots of accepted draws for population parameters.
for parameter_index in eachindex(true_individual_parameters)
    figure = plot(1:(n_draws), individual_parameter_draws[:,parameter_index],
    label = "accepted draws", legend=:right)
    plot!([0, n_draws],[true_individual_parameters[parameter_index], true_individual_parameters[parameter_index]],
    label = "true parameter value")
    title!("$(parameter_labels_all[parameter_index]) accepted MCMC draws for individual $(individual_labels[parameter_index])")
    plot!([0],[0], label="")
    display(figure)
end