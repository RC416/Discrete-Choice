
using Distributions, LinearAlgebra, ForwardDiff, FiniteDiff, Optim,
        DataFrames, BenchmarkTools, Profile, Plots
include("data structs.jl")
include("functions.jl")

# ----------------------------------------------------------------------
# 1. Model parameters and environment.
# ----------------------------------------------------------------------
# Utility parameter distributions.
β_mean = [0.0, 0.5, -0.5]
β_var = [2.0]

# Utility function
u(xᵢⱼ, βᵢ) = xᵢⱼ'βᵢ

# Environment.
n_available_choices = 4

# Dataset parameters.
n_individuals = 20000
n_individual_choices = 3

# Simulation parameters for estimation.
n_simulations = 200
β_draws = rand(Normal(0,1), (length(β_mean), n_individuals, n_simulations))

# ----------------------------------------------------------------------
# 2. Data structs and preallocated arrays.
# ----------------------------------------------------------------------
Model_Parameters = Model_Parameters_Struct(β_mean, β_var)
Environment = Environment_Struct(n_available_choices, n_individuals, n_individual_choices,
                                                             n_simulations, β_draws)

# ----------------------------------------------------------------------
# 3. Simulate data.
# ----------------------------------------------------------------------
X, Choices = simulate_dataset(Model_Parameters, Environment)
#X, Choices = load__real_dataset()
Data = Data_Struct(X, Choices)

# ----------------------------------------------------------------------
# 4. Estimate the model.
# ----------------------------------------------------------------------
guess = [zeros(length(β_mean)); ones(length(β_var))]
guess = [β_mean; β_var]

solution = optimize(likelihood_objective, guess, Newton(), autodiff = :forward)
#solution = optimize(likelihood_objective, guess, NewtonTrustRegion(), autodiff = :forward) 
#solution = optimize(likelihood_objective, guess, Newton()) # Newton's method with numerical derivatives
#solution = optimize(likelihood_objective, guess, LBFGS()) # uses gradient
#solution = optimize(likelihood_objective, guess, BFGS()) # gradient-free

# Point estimates.
β = solution.minimizer

# Standard errors.
H = ForwardDiff.hessian(likelihood_objective, β)
SE = sqrt.(diag((inv(H))))

# ----------------------------------------------------------------------
# 5. Visualizations.
# ----------------------------------------------------------------------
true_values = [β_mean; β_var]

# Table of parameter estimates.
column_names = ["variable", "true value", "mean", "95% CI low", "95% CI high"]
variable_names = ["$parameter_field[$parameter_index]" for parameter_field in fieldnames(Model_Parameters_Struct)
                                        for parameter_index in eachindex(getfield(Model_Parameters,parameter_field))]
results = [variable_names, true_values, β, β .- 1.96*SE, β .+ 1.96*SE]
display(DataFrame(results, column_names))

# Plot the likelihood function around the true values.
plot_likelihood(true_values)

# ----------------------------------------------------------------------
# 6. Benchmarking and profiling.
# ----------------------------------------------------------------------
function repeat(function_to_profile, vector_of_inputs, n_repetitions)
    for _ in 1:n_repetitions
        function_to_profile(vector_of_inputs...)
    end
end

@btime likelihood(Model_Parameters, Data, Environment)
@profview repeat(likelihood_objective, [guess], 10)