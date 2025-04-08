import numpy as np
import matplotlib.pyplot as plt
from ..components.constant_pressure_fitness import ConstantPressureFitness

# Define ranges for reference values
temperature_range = np.linspace(800, 4000, 500)  # Temperature range (K)
reactions_range = np.linspace(50, 300, 500)  # Number of reactions range
ignition_delay_range = np.linspace(0.001, 0.02, 500)  # Ignition delay range (ms)

# Fixed reduced values
reduced_temperature = 2500  # Fixed reduced temperature (K)
reduced_reactions = 50  # Fixed reduced number of reactions
reduced_ignition_delay = 0.005  # Fixed reduced ignition delay (ms)

# Define sigma and lambda values for testing
sigma_values = [6, 6, 8, 8]  # Sharpening factors
lambda_values = [0.3, 0.5, 0.8, 1.0]  # Shift parameters for sigmoid_integral

# Initialize the plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # Temperature Fitness Plot
# for sigma in sigma_values:
#     fitness_calculator = ConstantPressureFitness(difference_function="logarithmic", sharpening_factor=sigma, lam=1)
#     temperature_fitness_values = [
#         fitness_calculator.calculate_difference_function(reduced_temperature, ref_temp)
#         for ref_temp in temperature_range
#     ]
#     # Normalize the fitness values
#     temperature_fitness_values = np.array(temperature_fitness_values) / max(temperature_fitness_values)
#     axes[0].plot(temperature_range, temperature_fitness_values, label=f"σ = {sigma}")

# axes[0].set_title("Normalized Temperature Fitness vs. Reference Temperature")
# axes[0].set_xlabel("Reference Temperature (K)")
# axes[0].set_ylabel("Normalized Fitness Value")
# axes[0].grid()
# axes[0].legend()

# Reactions Fitness Plot (using sigmoid_integral with varying lambda)
for sigma, lam in zip(sigma_values, lambda_values):
    
    fitness_calculator = ConstantPressureFitness(difference_function="sigmoid", sharpening_factor=sigma, lam=lam)
    reactions_fitness_values = [
        fitness_calculator.calculate_difference_function(reduced_reactions, ref_reactions)
        for ref_reactions in reactions_range
    ]
    # Normalize the fitness values
    reactions_fitness_values = np.array(reactions_fitness_values) / max(reactions_fitness_values)
    axes[1].plot(reactions_range, reactions_fitness_values, label=f"σ = {sigma}, λ = {lam}")

axes[1].set_title("Normalized Reactions Fitness vs. Reference Number of Reactions")
axes[1].set_xlabel("Reference Number of Reactions")
axes[1].set_ylabel("Normalized Fitness Value")
axes[1].grid()
axes[1].legend()

# # Ignition Delay Fitness Plot
# for sigma in sigma_values:
#     fitness_calculator = ConstantPressureFitness(difference_function="logarithmic", sharpening_factor=sigma, lam=1)
#     ignition_delay_fitness_values = [
#         fitness_calculator.calculate_difference_function(reduced_ignition_delay, ref_delay)
#         for ref_delay in ignition_delay_range
#     ]
#     # Normalize the fitness values
#     ignition_delay_fitness_values = np.array(ignition_delay_fitness_values) / max(ignition_delay_fitness_values)
#     axes[2].plot(ignition_delay_range, ignition_delay_fitness_values, label=f"σ = {sigma}")

# axes[2].set_title("Normalized Ignition Delay Fitness vs. Reference Ignition Delay")
# axes[2].set_xlabel("Reference Ignition Delay (ms)")
# axes[2].set_ylabel("Normalized Fitness Value")
# axes[2].grid()
# axes[2].legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()