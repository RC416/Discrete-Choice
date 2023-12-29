#=
Code and functions to create a simulated dataset.
Used in main.jl.
=#

# Model environment.
n_individuals = 200
n_periods = 20
n_choices = size(Q₀, 1)          # extra choice for the (last) product which has normalized priors

# Individual signal draws for experience and detailing signals used to crate the observables.
experience_signals = rand(Normal(0, sqrt(σ²ₑ[1])), (n_individuals, n_periods - 1, n_choices)) # note: no signals for last period
advertising_signals =  rand(Normal(0, sqrt(σ²ₐ[1])), (n_individuals, n_periods - 1, n_choices)) # note: no signals for last period

# Random standard signal draws for simulated likelihood.
experience_signals_simulated = rand(Normal(0, 1), (n_individuals, n_periods - 1, n_choices, n_simulations))
advertising_signals_simulated = rand(Normal(0, 1), (n_individuals, n_periods - 1, n_choices, n_simulations))

# -----------------------------------------------------------------------------
# 2. Simulate data.
# -----------------------------------------------------------------------------
# Simulate choice observations.
Choice_Count = Array{Int64, 3}(undef, (n_individuals, n_periods, n_choices))
Choice_Probabilities = Array{Float64, 3}(undef, (n_individuals, n_periods, n_choices))

# Observables derived from choices (experience) and advertising. 
dₑ = Array{Int64, 3}(undef, (n_individuals, n_periods - 1, n_choices))       # indicator for whether product j was chosen by i in t-1
Nₑ = Array{Int64, 3}(undef, (n_individuals, n_periods - 1, n_choices))       # sum of dₑ over all previous periods for i, j
dₐ = Array{Int64, 3}(undef, (n_individuals, n_periods - 1, n_choices))       # indicator for whether informative detailing was received by i in t-1
Nₐ = Array{Int64, 3}(undef, (n_individuals, n_periods - 1, n_choices))       # sum of dₐ over all previous periods for i, j

# Total number of choices in each period.
Total_Choice_Count = 1  # start with just a fixed value

# Simulate advertising observations.
Advertising = Array{Int64, 3}(undef, (n_individuals, n_periods - 1, n_choices))
Advertising = 1 .* (rand(Uniform(0,1), (n_individuals, n_periods, n_choices)) .> 0.40)

# Record detailing interactions in dₐ, Nₐ accordingly.
dₐ .= (Advertising[:,2:n_periods,:] .> 0)
Nₐ = cumsum(dₐ, dims = 2)
## Nₐ = cumsum(Advertising, dims = 2)

# Simulate product choices in each period based on choice probabilities. 
for i in 1:n_individuals
    for t in 1:n_periods

        # Calculate choice probabilities.
        v = zeros(n_choices)

        for j in 1:n_choices

            # Get current beliefs.
            if t == 1 # if this is the first period, get the priors.
                
                Qᵢₜⱼ = Q₀[j]
                σ²ᵢₜⱼ = σ²₀[j]
            
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
            v[j] = EU(Qᵢₜⱼ, σ²ᵢₜⱼ, σ²ₑ, σ²ₐ, utility_parameters)
        end

        # Calculate CCPs.
        choice_probs = exp.(v) ./ sum(exp.(v))

        # Select products according to CCPs.
        observed_choices = sample(1:n_choices, Weights(choice_probs), Total_Choice_Count)
        
        # Calculate observed choice probabilities.
        observed_choice_counts = zeros(n_choices)

        for choice in observed_choices
            for j in 1:n_choices
                if choice == j
                    observed_choice_counts[j] += 1
                end
            end
        end
        
        # Record observables.
        Choice_Count[i,t,:] = observed_choice_counts
        observed_choice_probs = observed_choice_counts ./ sum(observed_choice_counts)
        Choice_Probabilities[i,t,:] = deepcopy(observed_choice_probs)
        
        for j in 1:n_choices
            if t < n_periods # exclude signals in last period

                # Indicator for experience signals (product j was chosen).
                dₑ[i,t,j] = 1.0 * (Choice_Count[i,t,j] > 0)
               
                # Cumulative count of experience signals.
                if t > 1
                    Nₑ[i,t,j] = Nₑ[i,t-1,j] + dₑ[i,t,j]
                    ##Nₑ[i,t,j] = Nₑ[i,t-1,j] + Choice_Count[i,t,j]
                elseif t == 1
                    Nₑ[i,t,j] = dₑ[i,t,j]
                    ##Nₑ[i,t,j] = Choice_Count[i,t,j]
                end
            end
        end
    end
end