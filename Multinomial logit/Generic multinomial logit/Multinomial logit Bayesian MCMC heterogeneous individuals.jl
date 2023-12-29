#=
This code solves a basic multinomial logit model using Bayesian Markov Chain Monte Carlo.
It samples from the posterior distribution using Metropolis-Hastings within Gibbs sampling.
The algorithm and theory is described in sections 12.6 and 12.5 of Train 2009, Discrete 
Choice Methods with Simulation as "Hierarchical Bayes for Mixed Logit".

Brief Model Description
    - Multinomial logit choice model with heterogeneous parameters.
    - Product specific utility: Uⱼᵢ = X₁β₁ᵢ + X₂β₂ᵢ + X₃β₃ᵢ + ϵⱼᵢ.
    - Individual chooses product j such that Uⱼ > Uₖ ∀k!=j.
    - Individual parameters are distributed βₙ ~ MvNormal(b,W).
    - ϵⱼᵢ is type 1 extreme value.
    - No period or individual fixed effects for simplicity.

Outline
    1. Set up model and simulate data.
        - Choose individual utility parameters and model environment.
        - Simulate individual utility parameters.
        - Simulate individual product choices.
    2. Define key functions.
        - Likelihood function and prior distribution for individual draws.
        - Random walk proposal distribution for indiviudal draws.
    3. Estimate model by drawing from Bayesian posterior.
        - Set up estimation parameters, including the starting parameter values.
        - See section 12.6 or 12.6.1 of Train 2009 for a description of
        the algorithm.
    4. Display results.
        - Build a table of parameter estimates and credible intervals.
        - Plot accepted draws.
        - Plot likelihood functions.
    5. Benchmarking and profiling.
        - Time and profile the likelihood function (main bottlneck).
        - Timing and profiling of the main loop in section 3. can be
        performed by encapsulating the loop in a function. 

Additional Notes
    - The code runs in approximately 2.5 minutes and is well optimized at the cost
    of a small amout of clarity.
    - The acceptance rate is tuned using the random walk variance (ρ). It is very 
    sensitive to the model conditions.
    - Algorithm is not very sensitive to the starting parameter values. 
=#

using LinearAlgebra
using Distributions
using DataFrames
using Plots
using Dates

# --------------------------------------------------------------------------------
# 1. Set up model and simulate data.
# --------------------------------------------------------------------------------
# Environment.
n_individuals = 100
n_periods = 100
n_products = 25

# True parameters.
b = [1.0, 2.0, -3.0]
W = diagm([0.5, 1.0, 1.5])
βₙ = rand(MvNormal(b,W), n_individuals)'

# Simulate dataset.
Y = Array{Int}(undef, (n_individuals, n_periods))               # observed choices
X = rand(Normal(0,1), n_products, size(b,1))                    # product characteristics
ϵ = rand(Gumbel(0,1), (n_individuals, n_periods, n_products))   # type 1 EV errors

# Get simulated choices.
for n in 1:n_individuals
    for t in 1:n_periods

        # Product-specific utilities.
        U = zeros(n_products)

        for j in 1:n_products
            U[j] = X[j,:] ⋅ βₙ[n,:] + ϵ[n,t,j]
        end

        # Record product with highest utility.
        Y[n,t] = argmax(U)
    end
end


# --------------------------------------------------------------------------------
# 2. Define key functions.
# --------------------------------------------------------------------------------
# Sample likelihood function.
function Log_Sample_Likelihood(Y, βₙ, X)

    # Get function parameters.
    log_likelihood = 0.0
    n_individuals, n_periods = size(Y)
    n_products = size(X, 1)
    U = zeros(n_products)

    for n in 1:n_individuals
        # Get individual's β values.
        βᵢ = @views βₙ[n,:]

        for t in 1:n_periods
            
            # Variables for logit formula (top and bottom).
            exp_Xⱼ = 0.0
            ∑ᵢexp_Xᵢ = 0.0
            
            # Get utility for each product choice.
            for j in 1:n_products
                                
                Uⱼ = 0.0
                for k in eachindex(βᵢ)
                    Uⱼ += X[j,k] * βᵢ[k]  # Uⱼ = X'β
                end
                #Uⱼ = @views X[j,:] ⋅ βᵢ # equivalent calculation, but slower 
    
                # Add to logit formula.
                exp_Uⱼ = exp(Uⱼ)
                ∑ᵢexp_Xᵢ += exp_Uⱼ                      # add all products to denominator
                if j == Y[n,t]; exp_Xⱼ += exp_Uⱼ end    # add chosen product to numerator
            end

            log_likelihood += log(exp_Xⱼ / ∑ᵢexp_Xᵢ)
            #log_likelihood += log(exp_Xⱼ) - log(∑ᵢexp_Xᵢ) # same speed
        end
    end
    return log_likelihood
end

# Individual likelihood function.
function Log_Sample_Likelihood(Y, βₙ, X, individual_index)

    # Get function parameters.
    log_likelihood = 0.0
    n_individuals, n_periods = size(Y)
    n_products = size(X, 1)
    U = zeros(n_products)

    # Get individual's β values.
    βᵢ = @views βₙ[individual_index, :]
    
    for t in 1:n_periods
        
        # Variables for logit formula (top and bottom).
        exp_Xⱼ = 0.0
        ∑ᵢexp_Xᵢ = 0.0
        
        # Get utility for each product choice.
        for j in 1:n_products

            Uⱼ = 0.0
            for k in eachindex(βᵢ)
                Uⱼ += X[j,k] * βᵢ[k]  # Uⱼ = X'β
            end
            #Uⱼ = @views X[j,:] ⋅ βᵢ # equivalent calculation, but slower 

            # Add to logit formula.
            exp_Uⱼ = exp(Uⱼ)
            ∑ᵢexp_Xᵢ += exp_Uⱼ                      # add all products to denominator
            if j == Y[individual_index, t]
                exp_Xⱼ += exp_Uⱼ                    # add chosen product to numerator
            end    
        end

        #log_likelihood += log(exp_Xⱼ / ∑ᵢexp_Xᵢ)
        log_likelihood += log(exp_Xⱼ) - log(∑ᵢexp_Xᵢ)
    end
    return log_likelihood
end

# Prior for βₙ.
function Log_Prior(βₙ, b, W, individual_index)

    # Get individual's β values.
    βᵢ = @views βₙ[individual_index, :]

    return logpdf(MvNormal(b,W), βᵢ)
end

# Proposal distribution for individual βₙ.
function Get_Proposal(βₙ, b, W, individual_index, ρ)

    # Get individual's β values.
    βᵢ = @views βₙ[individual_index,:]

    return βᵢ .+ rand(MvNormal(zeros(size(βᵢ)), W*ρ^2))
end


# --------------------------------------------------------------------------------
# 3. Estimate model using Gibbs Sampling and MCMC.
# --------------------------------------------------------------------------------
# Estimation parameters.
n_draws = 30000      # number of total draws
burn_in = 10000      # number of initial draws to ignore
ρ = 0.300            # random walk variance
individual_proposal_counts = zeros(Int, n_individuals)

# Arrays to store accepted draws.
b_draws = Array{Float64}(undef, (n_draws, size(b,1)))
W_draws = Array{Float64}(undef, (n_draws, size(W,1), size(W,2)))
βₙ_draws = Array{Float64}(undef, (n_draws, n_individuals, size(b,1)))

# Parameter guess values.
b_draws[1,:] = ones(size(b))
W_draws[1,:,:] = diagm(ones(size(b)))
βₙ_draws[1,:,:] = rand(MvNormal(b_draws[1,:], W_draws[1,:,:]), n_individuals)'

# Main loop.
start_time = now()

for d in 2:n_draws

    # Step 1. Draw b through Gibbs sampling.
    b̄ = vec(mean(βₙ_draws[d-1,:,:], dims=1))
    W̄ = W_draws[d-1,:,:] ./ n_individuals
    b_draws[d,:] = rand(MvNormal(b̄, W̄))

    # Step 2. Draw W through Gibbs sampling.
    p1 = size(b,1) + n_individuals
    S = zeros(size(b,1), size(b,1))
    for βₙ_ind in eachrow(βₙ_draws[d-1,:,:])
        S += (βₙ_ind - b_draws[d,:]) * (βₙ_ind - b_draws[d,:])' ./ n_individuals
    end    
    p2 = (size(b,1)*I(size(b,1)) .+ n_individuals*S) / (size(b,1) + n_individuals)
    W_draws[d,:,:] = rand(InverseWishart(p1, p2)) * n_individuals

    # Step 3. Draw βₙ through Metropolis-Hastings.
    Threads.@threads for n in 1:n_individuals
    #for n in 1:n_individuals

        # Get log likelhood and log prior for the previous draw.
        Log_Likelihood_prevous = Log_Sample_Likelihood(Y, βₙ_draws[d-1,:,:], X, n)
        Log_Prior_previous = Log_Prior(βₙ_draws[d-1,:,:], b_draws[d,:], W_draws[d,:,:], n)

        # Search over propsal draws.
        accepted_draw = false
        proposal_count = 0

        while(accepted_draw == false)

            proposal_count += 1
            if (proposal_count > 500) && (d > burn_in)
                println("too many proposals: $n")
                break
            end

            # Get proposed draw.
            βₙ_draws[d,n,:] = Get_Proposal(βₙ_draws[d-1,:,:], b_draws[d,:], W_draws[d,:,:], n, ρ)

            # Calculate likelihoods.
            Log_Likelihood_proposed = Log_Sample_Likelihood(Y, βₙ_draws[d,:,:], X, n)
            Log_Prior_proposed = Log_Prior(βₙ_draws[d,:,:], b_draws[d,:], W_draws[d,:,:], n)

            # Acceptance ratio.
            r = exp(Log_Likelihood_proposed + Log_Prior_proposed 
                    - Log_Likelihood_prevous - Log_Prior_previous)

            # Decide whether to accept.
            if (r > 1.0) | (r > rand(Uniform(0,1)))
                accepted_draw = true
                individual_proposal_counts[n] += proposal_count
            end
        end
    end

    # Display progress.
    if (mod(d, 500) == 0) | (d == 10); 
        println("completed $d or $n_draws draws, acceptance rates: 
        $(round.((1 ./ (individual_proposal_counts ./ d))[1:10], sigdigits=2)) \n")
    end
end


# --------------------------------------------------------------------------------
# 4. Display results.
# --------------------------------------------------------------------------------
# Display total compute time.
end_time = now()
println(canonicalize(Dates.CompoundPeriod(DateTime(end_time) - DateTime(start_time))))

# Table of parameter estiamtes.
raw_draws = [b_draws W_draws[:,1,1] W_draws[:,2,2] W_draws[:,3,3] W_draws[:,1,3] βₙ_draws[:,1,:]  βₙ_draws[:,10,:]]
raw_draws = raw_draws[burn_in:n_draws, :] # drop draws during burn-in period
#raw_draws = raw_draws_no_burn_in[        # keep only every 10th draw to avoid correlation  
#    [n for n in 1:size(raw_draws,1) if mod(n,10) == 0], :] 
true_values = [b; diag(W); W[1,3]; βₙ[1,:]; βₙ[10,:]]
mean_observed = vec(mean(raw_draws, dims=1))
bottom_95_CI = sort(raw_draws, dims=1)[Int(round(0.025*size(raw_draws, 1))), :]
top_95_CI = sort(raw_draws, dims=1)[Int(round(0.975*size(raw_draws, 1))), :]
bayesian_IJC_MCMC_results = [true_values mean_observed bottom_95_CI top_95_CI]
variable_names = ["b₁", "b₂", "b₃", "W₁₁", "W₂₂", "W₃₃", "W₁₃", "β₁-1", "β₂-1", "β₃-1", "β₁-10", "β₂-10", "β₃-10"]
display(DataFrame([variable_names bayesian_IJC_MCMC_results],
                  ["variable", "true value", "estimated mean", 
                  "95% Credible Interval Low", "95% Credible Interval High"]))

# Plots of accepted draws.
for var in eachindex(raw_draws[1,:])
    figure = plot(1:size(raw_draws,1), raw_draws[:,var], label="accepted $(variable_names[var]) draws")
    plot!([0.0, size(raw_draws,1)], [true_values[var], true_values[var]], label="true value", lines=:dash)
    if true_values[var] > 0.0; plot!([0.0],[0.0], legend=:bottomright, label="") end
    if true_values[var] <= 0.0; plot!([0.0],[0.0], legend=:topright, label="") end
    ylabel!(variable_names[var])
    display(figure)
end

# Plot the likelihood function.
b_range = Base._linspace(-5.00, 5.00, 100)
Likelihood_at_β = zeros(n_individuals, size(b,1), size(b_range,1))
individuals = [1,10]

for i in individuals
    for n in eachindex(b_range)
        Likelihood_at_β[i,1,n] = Log_Sample_Likelihood([Y[i,:]'; Y], [[b_range[n], βₙ[i,2], βₙ[i,3]]'; βₙ], X, 1)
        Likelihood_at_β[i,2,n] = Log_Sample_Likelihood([Y[i,:]'; Y], [[βₙ[i,1], b_range[n], βₙ[i,3]]'; βₙ], X, 1)
        Likelihood_at_β[i,3,n] = Log_Sample_Likelihood([Y[i,:]'; Y], [[βₙ[i,1], βₙ[i,2], b_range[n]]'; βₙ], X, 1)
    end
end

# Create likelihood plots.
for i in individuals
    for bᵢ in eachindex(b)
        figure = plot(b_range, Likelihood_at_β[i,bᵢ,:], label="L(Y|θ)", legend=:bottomright)
        xlabel!("β")
        ylabel!("L(Y|θ)")
        title!("Likelihood for Individual $i's β$bᵢ parameter")
        plot!([βₙ[i,bᵢ], βₙ[i,bᵢ]], [Likelihood_at_β[i,bᵢ,:][argmax(-Likelihood_at_β[i,bᵢ,:])],
         Likelihood_at_β[i,bᵢ,:][argmax(Likelihood_at_β[i,bᵢ,:])]], label="true value", line=:dash)
        plot!([b[bᵢ], b[bᵢ]], [Likelihood_at_β[i,bᵢ,:][argmax(-Likelihood_at_β[i,bᵢ,:])],
         Likelihood_at_β[i,bᵢ,:][argmax(Likelihood_at_β[i,bᵢ,:])]], label="population mean", line=:dash, color=:black)
        display(figure)
    end
end


# --------------------------------------------------------------------------------
# 5. Benchmarking and profiling.
# --------------------------------------------------------------------------------
using Profile
using BenchmarkTools

@btime Log_Sample_Likelihood(Y, βₙ, X)
@btime Log_Sample_Likelihood(Y, βₙ, X, 1)

function profile_Log_Sample_Likelihood(N)
    for n in 1:N
        Log_Sample_Likelihood(Y, βₙ, X)
    end
end
@profview profile_Log_Sample_Likelihood(100)
