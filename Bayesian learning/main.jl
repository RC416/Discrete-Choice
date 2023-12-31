#=
Bayesian learning model.
See documentation.
=#

using Distributions, StatsBase, Parameters, ForwardDiff, LinearAlgebra, Plots, DataFrames, Base

# -----------------------------------------------------------------------------
# 1. Model parameters.
# -----------------------------------------------------------------------------
utility_parameters = [-0.05]            # risk aversion parameter
true_quality = [0.0, 0.0, 0.0]      # true product quality 
Q₀  = [-1.0, -1.0, -1.0]               # product quality belief prior means normal distributed
σ²₀ = [0.50, 0.50, 0.50]                # mean parameters for distribution of individual variance beliefs (̂σⱼ₀)
σ²ₑ = [0.50]                            # variance of experience signals
σ²ₐ = [0.50]                            # variance of advertising signals

U(Qᵢₜⱼ, θ) = Qᵢₜⱼ + θ[1]*(Qᵢₜⱼ)^2  # utility function (quadratic)
EU(Qᵢₜⱼ, σ²ᵢₜⱼ, σ²ₑ, σ²ₐ, θ) = Qᵢₜⱼ + θ[1]*(Qᵢₜⱼ)^2 + θ[1]*(σ²ᵢₜⱼ + σ²ₑ[1])  # expected utility

n_simulations = 50                      # number of draws for simulated likelihood


# -----------------------------------------------------------------------------
# 2. Data and Environment.
# -----------------------------------------------------------------------------
# Add functions and struct definitions.
include("functions and structs.jl")

# Simulate data based on given parameters.
include("simulate data.jl")

# Initialize structs.
Model_Parameters = Model_Parameters_Struct(utility_parameters, true_quality, 
                    Q₀, σ²₀, σ²ₑ, σ²ₐ)
Environment = Environment_Struct(n_individuals, n_periods, n_choices,
        n_simulations, experience_signals_simulated, advertising_signals_simulated)
Data = Data_Struct(Choice_Count, Choice_Probabilities, Advertising, dₑ, dₐ, Nₑ, Nₐ)

# Optional: visualize the likelihood function around the true values
plot_likelihood_function(Model_Parameters) 


# -----------------------------------------------------------------------------
# 3. - Maximum likelihood estimation (manual).
# -----------------------------------------------------------------------------
# Guess parameter values.
true_values = Model_Parameters
free_parameter_indices = [2,3,5,6,7,8,9,10,12]
vector_true_values = struct_to_vector(true_values)[free_parameter_indices]

vector_guess = vector_true_values
θₙ = vector_guess
H = likelihood_hessian(θₙ)
eigen(H)

# θₙ = vector_guess
θₙ₊₁ = similar(θₙ)

# Search parameters.
distance = Inf
distance_target = 1e-4
iteration_count = 0
max_iterations = 500
stop_condition_met = false
converged = false
step_size = 0.1

# Main loop.
while(stop_condition_met == false)

    # Get next set of parameter values.
    ∇ = likelihood_gradient(θₙ)
    H = likelihood_hessian(θₙ) - norm(∇) * I(size(θₙ,1))

    # Line search over step sizes.
    θₙ₊₁ = θₙ - step_size * inv(H) * ∇
    best_step = likelihood_objective(θₙ₊₁)

    for smaller_step in [step_size / 2^n for n in 1:1]
        θₙ₊₁_candidate = θₙ - smaller_step * inv(H) * ∇
        candidate_step = likelihood_objective(θₙ₊₁_candidate)

        if candidate_step > best_step
            θₙ₊₁ = θₙ₊₁_candidate
        else
            break
        end
    end

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
for field in fieldnames(Model_Parameters_Struct)
    values = getfield(Model_Parameters, field)
    for index in eachindex(values)
        push!(parameter_labels, "$field index $index")
    end
end
parameter_labels = parameter_labels[free_parameter_indices] # drop fixed parameters

# Report results.
results = [θₙ θₙ .- 1.96*SE θₙ .+ 1.96*SE]
println("\n Maximum Likelihood Parameter Estimates")
display(DataFrame([parameter_labels vector_guess vector_true_values results],
                  ["variable", "guess", "true value", "estimated mean", 
                  "95% Confidence Interval Low", "95% Confidence Interval High"]))


# -----------------------------------------------------------------------------
# 4. - Maximum likelihood estimation (Optim package).
# -----------------------------------------------------------------------------
using Optim

# Set up parameter values.
free_parameter_indices = [2,3,5,6,7,8,9,10,12]
true_values = Model_Parameters
vector_true_values = struct_to_vector(true_values)[free_parameter_indices]
vector_guess = vector_true_values

optim_objective(x) = -likelihood_objective(x)
opt_options = Optim.Options(
    show_trace = true,
    show_every = 1,
    iterations=100)

result = optimize(optim_objective, vector_guess, NewtonTrustRegion(), autodiff = :forward, opt_options)

# Report results.
θ = result.minimizer
H = likelihood_hessian(θ)
SE = sqrt.(diag(inv(-H)))

# Parameter labels.
parameter_labels = []
for field in fieldnames(Model_Parameters_Struct)
    values = getfield(Model_Parameters, field)
    for index in eachindex(values)
        push!(parameter_labels, "$field index $index")
    end
end
parameter_labels = parameter_labels[free_parameter_indices] # drop fixed parameters

# Results table.
results = [θ   θ .- 1.96*SE   θ .+ 1.96*SE]
println("\n Maximum Likelihood Parameter Estimates")
display(DataFrame([parameter_labels vector_guess results],
                  ["variable", "guess", "estimated mean", 
                  "95% Confidence Interval Low", "95% Confidence Interval High"]))