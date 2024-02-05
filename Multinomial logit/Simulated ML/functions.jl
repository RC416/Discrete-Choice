include("data structs.jl")

# Simulate dataset.
function simulate_dataset(Model_Parameters, Environment)

    # Unpack relevant parameters.
    @unpack β_mean, β_var = Model_Parameters
    @unpack n_available_choices, n_individuals, n_individual_choices = Environment

    # Individual utility parameters.
    normal_draws = rand(Normal(0,1), (length(β_mean), n_individuals))
    β = β_mean .+ (normal_draws .* sqrt.(β_var))
    # β = rand(MvNormal(β_mean, diagm(β_var)), n_individuals) # equivalent

    # Observables.
    X = rand(Uniform(0,1), (length(β_mean), n_available_choices, n_individuals, n_individual_choices))

    # T1EV unobservable errors.
    ϵ = rand(Gumbel(0,1), (n_available_choices, n_individuals, n_individual_choices))

    # Get observed choices.
    Choices = Matrix{Int64}(undef, n_individuals, n_individual_choices)
    vᵢ = Vector{Float64}(undef, n_available_choices)

    for i in 1:n_individuals
        for t in 1:n_individual_choices   
            for j in 1:n_available_choices

                # Get observables and unobservables.
                Xᵢⱼ = @view X[:,j,i,t] # @view causes the array slice to be a view instead of a copy
                βᵢ = @view β[:,i]
                ϵᵢⱼ = ϵ[j,i,t]

                # Calculate utility for this choice.
                vᵢ[j] = u(Xᵢⱼ, βᵢ) + ϵᵢⱼ
            end

            # Get the optimal choice.
            Choices[i, t] = argmax(vᵢ)
        end
    end

    return X, Choices
end

# Sample log likelihood.
function likelihood(Model_Parameters, Data, Environment)

    # Unpack relevant parameters.
    @unpack β_mean, β_var = Model_Parameters
    @unpack X, Choices = Data
    @unpack n_individuals, n_individual_choices, n_available_choices, 
                                                n_simulations, β_draws = Environment

    # Adjust the fixed simulated individual draws with given population parameters.
    β = (β_mean .+ (β_draws .* sqrt.(β_var)))
    #β = rand(MvNormal(β_mean, β_var), n_simulations)

    # Preallocate variables. Remain generic for parameter type to enable autodiff.
    parameter_value_type = eltype(β_mean)
    exp_vᵢ = Vector{parameter_value_type}(undef, n_available_choices)
    CCP = Matrix{parameter_value_type}(undef, n_available_choices, n_simulations)
    ℓ = zero(parameter_value_type) # = 0.0
    
    # Calculate likelihood
    for i in 1:n_individuals
        for t in 1:n_individual_choices

            ℓᵢ = 0.0

            for n in 1:n_simulations

                # Get indirect utility for each choice.
                for j in 1:n_available_choices

                    # Get observables and unobservables.
                    Xᵢⱼ = @views X[:,j,i,t] # @view causes the array slice to be a view instead of a copy
                    βᵢ = @views β[:,i,n]

                    # Calculate utility for this choice.
                    exp_vᵢ[j] = exp(u(Xᵢⱼ, βᵢ))
                end

                # Calculate CCPs for this simulation.
                CCP[:,n] .= exp_vᵢ ./ sum(exp_vᵢ)
            end
            
            # Calculate the simulated choice probability.
            CCP_ave = mean(CCP, dims=2)

            # Record the simulated likelhood.
            j_chosen = Choices[i,t]
            ℓ += log(CCP_ave[j_chosen])
        end
    end

    return ℓ
end

# Functions for optimization.
function likelihood_objective(parameter_vector)
    
    Model_Parameters = Model_Parameters_Struct(parameter_vector[1:length(β_mean)],
    parameter_vector[1+length(β_mean):lastindex(parameter_vector)])

    return -likelihood(Model_Parameters, Data, Environment)
end

function likelihood_gradient!(∇, parameter_vector)
    ∇ .= ForwardDiff.gradient(likelihood_objective, parameter_vector)
end

function likelihood_hessian!(H, parameter_vector)
    H .= ForwardDiff.hessian(likelihood_objective, parameter_vector)
end

# Visualization functions.
function plot_likelihood(true_values)

    for parameter_index in eachindex(true_values)

        # Create range of parameter values around the true value.
        parameter_values = deepcopy(true_values)
        true_value = true_values[parameter_index]
        x_range = LinRange(true_value - 1.0, true_value + 1.0, 10)
        y = similar(x_range)

        # Get the likelhood on range of parameter values.
        for x_index in eachindex(x_range)
            parameter_values[parameter_index] = x_range[x_index]
            y[x_index] = likelihood_objective(parameter_values)
        end

        # Create plot.
        fig = plot(x_range, y, title="Parameter $parameter_index, true value = $true_value", label = "")
        plot!([true_value, true_value], [minimum(y), maximum(y)], linestyle = :dash, color = :black, label = "true value")
        display(fig)
    end
end