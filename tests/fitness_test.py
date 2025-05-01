import numpy as np
import cantera as ct
import math
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import pandas as pd

class ConstantPressureFitness:
    def __init__(self, difference_function, sharpening_factor, normalization_method, condition=None, weights=None):
        self.difference_function = difference_function
        self.sharpening_factor = sharpening_factor
        self.normalization_method = normalization_method  # "sigmoid" or "logarithmic"

        self.condition = condition if condition is not None else {}
        self.temperature = self.condition.get('temperature', None)
        self.weights = weights if weights is not None else {}

    def calculate_difference_function(self, reduced, full):
        """Calculate the difference between reduced and full values using the specified method."""
        try:
            if self.difference_function == "linear":
                return abs(reduced - full)

            elif self.difference_function == "squared":
                return (reduced - full) ** 2

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

    def normalize(self, raw_fitness):
        """Normalize the raw fitness value using the specified normalization method."""
        try:
            if self.normalization_method == "sigmoid":
                # return 1 / (1 + math.exp(self.sharpening_factor * (1- raw_fitness )))
                return (1 * (1 / (1 + math.exp(-self.sharpening_factor * (raw_fitness - 0.5)))))

            elif self.normalization_method == "logarithmic":
                return math.log(1 + self.sharpening_factor * abs(raw_fitness))

            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        except Exception as e:
            print(f"Error in normalize: {e}")
            return float("inf")

    def calculate_error(self, time, reduced, full):
        """Calculate the error using the integral-based approach."""
        try:
            # Apply the difference function element-wise
            differences = np.array([self.calculate_difference_function(r, f) for r, f in zip(reduced, full)])
            
            if self.difference_function == "linear":
                denominators = np.abs(full)  # Absolute value of the reference profile
            elif self.difference_function == "squared":
                denominators = full ** 2  # Squared value of the reference profile
            else:
                raise ValueError("Unsupported difference function for denominator calculation.")

            # Perform numerical integration
            numerator = simpson(differences)
            denominator = simpson(denominators)

            if denominator == 0:
                return float("inf")

            # Calculate raw fitness
            raw_fitness = numerator / denominator

            # Normalize the fitness
            return raw_fitness
        except Exception as e:
            print(f"Error in calculate_error: {e}")
            return float("inf")

    def temperature_fitness(self, reduced_profiles, full_profiles):
        """Calculate fitness based on temperature profile."""
        try:
            time = full_profiles["time"]
            temp_red = reduced_profiles["temperature_profile"]
            temp_full = full_profiles["temperature_profile"]
            raw_fitness = self.calculate_error(time, temp_red, temp_full)
            # normalized_fitness = self.normalize(raw_fitness)
            # Log raw and normalized fitness
            # print(f"Temperature Fitness - Raw Fitness: {raw_fitness}, Normalized Fitness: {normalized_fitness}")
            return raw_fitness
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
            raw_fitness = 0.0

            for species in key_species:
                if species not in reduced_mole_fractions:
                    print(f"Warning: Species '{species}' is missing in the reduced mechanism. Skipping.")
                    continue

                species_weight = self.weights["species"].get(species, 0.0)  # Default weight is 0.0 if not specified
                xi_red = reduced_profiles["mole_fractions"][species]
                xi_full = full_profiles["mole_fractions"][species]
                
                # Calculate error for the species
                raw_fitness += self.calculate_error(time, xi_red, xi_full)
                # normalized_fitness = self.normalize(raw_fitness)
                fitness += raw_fitness * species_weight
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
            raw_fitness = self.calculate_difference_function(reduced_delay, full_delay)
            # normalized_fitness = self.normalize(raw_fitness)
            return raw_fitness
        except Exception as e:
            print(f"Error in ignition_delay_fitness: {e}")
            return float("inf")
    
    def reaction_count_fitness(self, reduced_reactions, full_reactions):
        """Calculate fitness based on the number of reactions."""
        try:
        # Calculate the raw fitness based on the selected difference function
            if self.difference_function == "linear":
                raw_fitness = abs(reduced_reactions) / full_reactions

            elif self.difference_function == "squared":
                raw_fitness = ((reduced_reactions) / full_reactions) ** 2
            else:
                raise ValueError(f"Unsupported difference function: {self.difference_function}")

            # Normalize the fitness using sigmoid scaling
            # normalized_fitness = self.normalize(raw_fitness)
            return raw_fitness
        except Exception as e:
            print(f"Error in reaction_count_fitness: {e}")
            return float("inf")
        
    def combined_fitness(self, reduced_profiles, full_profiles, key_species, reduced_reactions, genome_length):
        """Calculate combined fitness using weighted components."""
        try:
            temp_fitness_raw = self.temperature_fitness(reduced_profiles, full_profiles) 
            species_fitness_raw = self.species_fitness(reduced_profiles, full_profiles, key_species)
            ignition_fitness_raw = self.ignition_delay_fitness(reduced_profiles, full_profiles)             
            reaction_fitness_raw = self.reaction_count_fitness(reduced_reactions, genome_length) 
            
            temp_fitness_norm = self.normalize(temp_fitness_raw)
            species_fitness_norm = self.normalize(species_fitness_raw)
            ignition_fitness_norm = self.normalize(ignition_fitness_raw)
            # reaction_fitness_norm = self.normalize(reaction_fitness_raw)
            
            temp_fitness = temp_fitness_norm * self.weights["temperature"]
            species_fitness = species_fitness_norm
            ignition_fitness = ignition_fitness_norm * self.weights["IDT"]
            reaction_fitness = reaction_fitness_raw * self.weights["reactions"] 
                
            
            total_weight = (self.weights["temperature"] + 
                            self.weights["IDT"] + 
                            sum(self.weights["species"].values())
                            )

            combined_raw_fitness = (temp_fitness_raw + 
                                    ignition_fitness_raw + 
                                    species_fitness_raw 
                                    ) 
            
            # combined_fitness_norm = self.normalize(combined_raw_fitness)
            
            combined_fitness = ((temp_fitness + species_fitness + ignition_fitness + reaction_fitness ))
            
            # combined_fitness = combined_fitness_norm + reaction_fitness
            
            # print("Raw combined fitness: ", combined_raw_fitness)
            # print("Norm combined fitness: ", total_fitness)
            # print("total_fitness_threee", total_fitness)
            
            return {
                "combined_fitness": combined_fitness,

                "temperature_fitness": temp_fitness,
                "species_fitness": species_fitness,
                "ignition_delay_fitness": ignition_fitness,
                "reaction_count_fitness": reaction_fitness
                }
        except Exception as e:
            print(f"Error in combined_fitness: {e}")
            return {
                "combined_fitness": float("inf"),
                "temperature_fitness": float("inf"),
                "species_fitness": float("inf"),
                "ignition_delay_fitness": float("inf"),
                "reaction_count_fitness": float("inf")
            }


# Reduced and full profiles
reduced_profiles = {
    "time": np.linspace(0, 0.05, 1000),
    "temperature_profile": np.linspace(1800, 3000, 1000) * 0.95,  # Reduced profile is 95% of full
    "mole_fractions": {
        "CH4": np.linspace(0.1, 0.01, 1000) * 0.9,  # Reduced profile is 90% of full
        "O2": np.linspace(0.21, 0.15, 1000) * 0.92,  # Reduced profile is 92% of full
        "CO2": np.linspace(0.05, 0.1, 1000) * 0.88,  # Reduced profile is 88% of full
        "H2O": np.linspace(0.02, 0.08, 1000) * 0.85,  # Reduced profile is 85% of full
    },
    "ignition_delay": 0.00254534*1000
}

full_profiles = {
    "time": np.linspace(0, 0.05, 1000),
    "temperature_profile": np.linspace(1800, 3000, 1000),
    "mole_fractions": {
        "CH4": np.linspace(0.1, 0.02, 1000),
        "O2": np.linspace(0.21, 0.15, 1000),
        "CO2": np.linspace(0.05, 0.1, 1000),
        "OH": np.linspace(0.02, 0.08, 1000),
    },
    "ignition_delay": 0.00168495*1000
}

# Weights
weights={
        "temperature": 1,
        "IDT": 1,
        "species": {"CH4": 0.2,
                    "O2": 0.3,
                    "CO2": 0.25,
                    "OH": 0.25
                },
        "reactions": 1,
    }

reduced_reactions = 150
full_reactions = 325

# Parameters
generations = 40
sigma_values = [1.0, 2.0, 4.0, 6.0, 8.0]
methods = ["squared"]
normalization_methods = ["sigmoid", "logarithmic"]

# Initialize the dataset
fitness_progression = []

# Simulate fitness progression for each sigma value and normalization method
# Simulate fitness progression for each sigma value and normalization method
for sigma in sigma_values:
    for method in methods:
        for norm_method in normalization_methods:
            fitness_calculator = ConstantPressureFitness(
                difference_function=method,
                sharpening_factor=sigma,
                normalization_method=norm_method,
                weights=weights
            )

            # Reset reduced profiles for each combination
            reduced_profiles = {
                "time": np.linspace(0, 0.05, 1000),
                "temperature_profile": np.copy(full_profiles["temperature_profile"]),  # Start identical to full
                "mole_fractions": {
                    species: np.copy(full_profiles["mole_fractions"][species])
                    for species in full_profiles["mole_fractions"]
                },
                "ignition_delay": full_profiles["ignition_delay"]
            }

            # Simulate fitness for multiple generations
            for generation in range(1, generations ):
                if generation > 1:  # Introduce divergence and improvement after the first generation
                    # Improvement factor: gradually approach the full profiles
                    improvement_factor = generation / (generations + 1)  # Increases with generations
                    noise_factor = 0.02 * (1 - improvement_factor)  # Add noise that decreases over generations

                    # Update reduced profiles to approach full profiles
                    reduced_profiles["temperature_profile"] = (
                        reduced_profiles["temperature_profile"] * (1 - improvement_factor) +
                        full_profiles["temperature_profile"] * improvement_factor +
                        np.random.normal(0, noise_factor, size=reduced_profiles["temperature_profile"].shape)
                    )
                    for species in reduced_profiles["mole_fractions"]:
                        reduced_profiles["mole_fractions"][species] = (
                            reduced_profiles["mole_fractions"][species] * (1 - improvement_factor) +
                            full_profiles["mole_fractions"][species] * improvement_factor +
                            np.random.normal(0, noise_factor, size=reduced_profiles["mole_fractions"][species].shape)
                        )
                    reduced_profiles["ignition_delay"] = (
                        reduced_profiles["ignition_delay"] * (1 - improvement_factor) +
                        full_profiles["ignition_delay"] * improvement_factor
                    )

                # Calculate fitness
                combined_fitness = fitness_calculator.combined_fitness(
                    reduced_profiles, full_profiles, key_species=["CH4", "O2", "CO2", "OH"],
                    reduced_reactions=reduced_reactions,
                    genome_length = full_reactions
                )

                # Append results to the dataset
                fitness_progression.append({
                    "Generation": generation,
                    "Sigma": sigma,
                    "Method": method,
                    "Normalization": norm_method,
                    "Combined Fitness": combined_fitness["combined_fitness"],
                    "Temperature Fitness": combined_fitness["temperature_fitness"],
                    "Species Fitness": combined_fitness["species_fitness"],
                    "Ignition Delay Fitness": combined_fitness["ignition_delay_fitness"],
                    "Reaction Count Fitness": combined_fitness["reaction_count_fitness"]
                })

# Convert to a DataFrame
fitness_df = pd.DataFrame(fitness_progression)

# Plot fitness progression for each combination of method and normalization
for method in methods:
    for norm_method in normalization_methods:
        plt.figure(figsize=(10, 6))
        for sigma in sigma_values:
            subset = fitness_df[
                (fitness_df["Method"] == method) &
                (fitness_df["Normalization"] == norm_method) &
                (fitness_df["Sigma"] == sigma)
            ]
            plt.plot(
                subset["Generation"],
                subset["Combined Fitness"],
                label=f"Sigma={sigma}"
            )

        plt.title(f"Fitness Progression ({method.capitalize()} + {norm_method.capitalize()})")
        plt.xlabel("Generation")
        plt.ylabel("Combined Fitness")
        plt.legend()
        plt.grid(alpha=0.7, linestyle="--")
        plt.show()