
import numpy as np
import cantera as ct
import math
from scipy.integrate import simpson
import matplotlib.pyplot as plt

class ConstantPressureFitness:
    def __init__(self, difference_function, sharpening_factor, lam, condition=None, weights=None):
        self.difference_function = difference_function
        self.sharpening_factor = sharpening_factor
        self.lam = lam

        self.condition = condition if condition is not None else {}
        self.temperature = self.condition.get('temperature', None)
        self.weights = weights if weights is not None else {}

    def calculate_difference_function(self, reduced, full):
        """Calculate the difference between reduced and full values using the specified method."""
        try:
            epsilon = 1e-8  # Small constant to avoid division by zero or log(0)
            full = max(full, epsilon)  # Ensure full is not zero
            reduced = max(reduced, epsilon)
            if self.difference_function == "linear":
                return abs(reduced - full)

            elif self.difference_function == "squared":
                return (reduced - full) ** 2

            elif self.difference_function == "logarithmic":
                if full == 0:
                    raise ValueError("full value is zero, cannot calculate logarithmic difference.")
                return math.log(1 + self.sharpening_factor * abs((reduced - full) / full))

            elif self.difference_function == "sigmoid":
                if full == 0:
                    raise ValueError("full value is zero, cannot calculate sigmoid difference.")
                return 1 / (1 + math.exp(self.sharpening_factor * (1 - (reduced / (self.lam * full)))))

        except ZeroDivisionError as e:
            print(f"Error: Division by zero encountered. {e}")
            return float("inf")
        except ValueError as e:
            print(f"Value Error: {e}")
            return float("inf")
        except OverflowError as e:
            print(f"Overflow Error: {e}")
            return float("inf")
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return float("inf")

    def calculate_error(self, time, reduced, full):
        """Calculate the error using the integral-based approach."""
        try:
            # Apply the difference function element-wise
            differences = np.array([self.calculate_difference_function(r, f) for r, f in zip(reduced, full)])

            # Perform numerical integration
            numerator = simpson(differences)
            denominator = simpson(full ** 2)

            if denominator == 0:
                return float("inf")
            return numerator / denominator
        except Exception as e:
            print(f"Error in calculate_error: {e}")
            return float("inf")

    def temperature_fitness(self, reduced_profiles, full_profiles):
        """Calculate fitness based on temperature profile."""
        try:
            time = full_profiles["time"]
            temp_red = reduced_profiles["temperature_profile"]
            temp_full = full_profiles["temperature_profile"]
            return self.calculate_error(time, temp_red, temp_full)
        except Exception as e:
            print(f"Error in temperature_fitness: {e}")
            return float("inf")

    def species_fitness(self, reduced_profiles, full_profiles, key_species):
        """Calculate fitness based on species mole fractions."""
        try:
            time = full_profiles["time"]
            reduced_mole_fractions = reduced_profiles["mole_fractions"]
            fitness = 0.0
            total_weight = 0.0

            for species in key_species:
                if species not in reduced_mole_fractions:
                    print(f"Warning: Species '{species}' is missing in the reduced mechanism. Skipping.")
                    continue

                species_weight = self.weights["species"].get(species, 0.0)  # Default weight is 0.0 if not specified
                xi_red = reduced_profiles["mole_fractions"][species]
                xi_full = full_profiles["mole_fractions"][species]

                # Calculate error for the species
                fitness += self.calculate_error(time, xi_red, xi_full) * species_weight
                total_weight += species_weight

            return fitness / total_weight if total_weight > 0 else float("inf")
        except Exception as e:
            print(f"Error in species_fitness: {e}")
            return float("inf")

    def ignition_delay_fitness(self, reduced_profiles, full_profiles):
        """Calculate fitness based on ignition delay time."""
        try:
            reduced_delay = reduced_profiles.get("ignition_delay", 0.0)
            full_delay = full_profiles.get("ignition_delay", 0.0)
            return self.calculate_difference_function(reduced_delay, full_delay)
        except Exception as e:
            print(f"Error in ignition_delay_fitness: {e}")
            return float("inf")

    def combined_fitness(self, reduced_profiles, full_profiles, key_species):
        """Calculate combined fitness using weighted components."""
        try:
            temp_fitness = self.temperature_fitness(reduced_profiles, full_profiles) * self.weights["temperature"]

            species_fitness = self.species_fitness(reduced_profiles, full_profiles, key_species)

            ignition_fitness = self.ignition_delay_fitness(reduced_profiles, full_profiles) * self.weights["IDT"]

            total_weight = self.weights["temperature"] + self.weights["IDT"] + sum(self.weights["species"].values())

            combined_fitness = (temp_fitness + ignition_fitness + species_fitness) / total_weight

            return {
                "combined_fitness": combined_fitness,
                "temperature_fitness": temp_fitness,
                "species_fitness": species_fitness,
                "ignition_delay_fitness": ignition_fitness
                }
        except Exception as e:
            print(f"Error in combined_fitness: {e}")
            return {
                "combined_fitness": float("inf"),
                "temperature_fitness": float("inf"),
                "species_fitness": float("inf"),
                "ignition_delay_fitness": float("inf")
            }
            
# Example profiles
reduced_profiles = {
    "time": np.linspace(0, 0.05, 1000),
    "temperature_profile": np.linspace(1736, 2815.59, 1000) * 0.95,  # Reduced profile is 95% of full
    "mole_fractions": {"CH4": np.linspace(0.1, 0.01, 1000) * 0.9}  # Reduced profile is 90% of full
}

full_profiles = {
    "time": np.linspace(0, 0.05, 1000),
    "temperature_profile": np.linspace(1736, 2823.32, 1000),
    "mole_fractions": {"CH4": np.linspace(0.1, 0.02, 1000)}
}

# Weights
weights = {
    "temperature": 1,
    "IDT": 1,
    "species": {"CH4": 0.2}
}

# Initialize the fitness calculator for different methods
methods = ["linear", "squared", "logarithmic", "sigmoid"]
fitness_results = {}

for method in methods:
    fitness_calculator = ConstantPressureFitness(
        difference_function=method,
        sharpening_factor=6.0,
        lam=0.1,
        weights=weights
    )

    # Calculate fitness
    combined_fitness = fitness_calculator.combined_fitness(reduced_profiles, full_profiles, key_species=["CH4"])
    fitness_results[method] = combined_fitness

# Plot the results
plt.figure(figsize=(10, 6))
x = np.arange(len(methods))
y = [fitness_results[method]["combined_fitness"] for method in methods]

plt.bar(x, y, color=["blue", "orange", "green", "red"])
plt.xticks(x, methods)
plt.ylabel("Combined Fitness")
plt.title("Comparison of Fitness Methods")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()            
 
 
            
#region
# class TestErrorCalculation:
#     def calculate_error(self, time, reduced, full):
#         """
#         Calculate the error between reduced and full mechanisms using Simpson's rule.
#         """
#         numerator = simpson(abs(reduced - full) ** 2)
#         denominator = simpson(full ** 2 )
#         # numerator = self.manual_integration(time, (reduced - full) ** 2)
#         # denominator = self.manual_integration(time, full ** 2 )
#         if denominator == 0:
#             return float("inf")
#         return (numerator / denominator)
    
#     def manual_integration(self, x, y):
#         integral = 0.0
#         for i in range(1, len(x)):
#             dx = x[i] - x[i - 1]
#             avg_height = (y[i] + y[i - 1]) / 2
#             integral += dx * avg_height
#         return integral
    
#     def temperature_error(self, reduced_profiles, full_profiles):

#         try:
#             # Extract time and temperature profiles
#             time = full_profiles["time"]
#             temp_red = reduced_profiles["temperature_profile"]
#             temp_full = full_profiles["temperature_profile"]

#             # Calculate the error using the integral-based approach
#             numerator = self.manual_integration(time, (temp_red - temp_full) ** 2)
#             denominator = self.manual_integration(time, temp_full ** 2)

#             # Handle the case where the denominator is zero
#             if denominator == 0:
#                 return float("inf")

#             # Return the normalized error
#             return numerator / denominator

#         except Exception as e:
#             print(f"Error in temperature_error: {e}")
#             return float("inf")
    
# # Create a test instance
# test_instance = TestErrorCalculation()

# # Test data
# time = np.linspace(0, 0.05, 1000)  # Time array from 0 to 0.05 seconds
# full = np.sin(2 * np.pi * time)  # Full mechanism data (e.g., sine wave)
# reduced = full * 0.9  # Reduced mechanism data (scaled version of full)

# # Call the function
# error = test_instance.calculate_error(time, reduced, full)

# # Print the result
# print(f"Calculated Error: {error}")

# reduced_profiles = {
#     "time": np.linspace(0, 0.05, 1000),
#     "temperature_profile": np.linspace(1736, 2815.59, 1000)  # Example reduced profile
# }

# full_profiles = {
#     "time": np.linspace(0, 0.05, 1000),
#     "temperature_profile": np.linspace(1736, 2823.32, 1000)  # Example full profile
# }

# # Calculate the temperature error
# error = test_instance.temperature_error(reduced_profiles, full_profiles)
# print(f"Temperature Error: {error}")
#endregion