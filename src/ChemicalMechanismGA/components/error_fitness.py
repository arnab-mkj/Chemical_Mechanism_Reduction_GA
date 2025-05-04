import numpy as np
import cantera as ct
import math
from scipy.integrate import simpson


class ConstantPressureFitness:
    def __init__(self, difference_function, sharpening_factor, normalization_method, condition=None, weights=None):
        """
        Initialize the fitness calculator for constant pressure combustion simulations.
        
        Args:
            difference_function (str): Method for calculating differences ("absolute" or "squared")
            sharpening_factor (float): Factor used in sigmoid normalization
            normalization_method (str): Normalization approach ("sigmoid" or "logarithmic")
            condition (dict, optional): Simulation conditions dictionary
            weights (dict, optional): Weights for different fitness components
        """
        self.difference_function = difference_function
        self.sharpening_factor = sharpening_factor
        self.normalization_method = normalization_method  # "sigmoid" or "logarithmic"
        self.condition = condition if condition is not None else {}
        self.weights = weights if weights is not None else {}

    def calculate_difference_function(self, reduced, full):
        """
        Calculate the difference between reduced and full values using the specified method.
        
        Args:
            reduced: Value from reduced mechanism
            full: Value from full mechanism
            
        Returns:
            float: Calculated difference or infinity if error occurs
        """
        try:
            if self.difference_function == "absolute":
                return np.sqrt((reduced - full)**2)
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
        """
        Normalize the raw fitness value using specified normalization method.
        
        Args:
            raw_fitness (float): Unnormalized fitness value
            
        Returns:
            float: Normalized fitness value or infinity if error occurs
        """
        try:
            if self.normalization_method == "sigmoid":
                # Sigmoid normalization: maps values to (0,1) range
                return float(1 / (1 + math.exp(-self.sharpening_factor * (raw_fitness))))
            elif self.normalization_method == "logarithmic":
                # Logarithmic normalization: compresses large value ranges
                return math.log(1 + 4 * abs(raw_fitness))
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        except Exception as e:
            print(f"Error in normalize: {e}")
            return float("inf")

    def calculate_error(self, time, reduced, full):
        """
        Calculate the error between profiles using numerical integration.
        
        Args:
            time (array): Time points for integration
            reduced (array): Profile from reduced mechanism
            full (array): Profile from full mechanism
            
        Returns:
            float: Integrated error metric or infinity if error occurs
        """
        try:
            # Calculate pointwise differences
            differences = np.array([self.calculate_difference_function(r, f) for r, f in zip(reduced, full)])
            
            # Determine denominators based on difference function
            if self.difference_function == "absolute":
                denominators = np.abs(full)  # Absolute value of reference
            elif self.difference_function == "squared":
                denominators = full ** 2
            else:
                raise ValueError("Unsupported difference function for denominator calculation.")

            # Numerical integration of differences and denominators
            numerator = simpson(differences)
            denominator = simpson(denominators)
            if denominator == 0:
                return float("inf")

            return numerator / denominator
        except Exception as e:
            print(f"Error in calculate_error: {e}")
            return float("inf")

    def temperature_fitness(self, reduced_profiles, full_profiles):
        """
        Calculate fitness based on temperature profile agreement.
        
        Args:
            reduced_profiles (dict): Results from reduced mechanism
            full_profiles (dict): Results from full mechanism
            
        Returns:
            float: Temperature fitness metric
        """
        try:
            time = full_profiles["time"]
            temp_red = reduced_profiles["temperature_profile"]
            temp_full = full_profiles["temperature_profile"]
            return self.calculate_error(time, temp_red, temp_full)
        except Exception as e:
            print(f"Error in temperature_fitness: {e}")
            return float("inf")

    def species_fitness(self, reduced_profiles, full_profiles, key_species):
        """
        Calculate fitness based on species mole fraction agreement.
        
        Args:
            reduced_profiles (dict): Results from reduced mechanism
            full_profiles (dict): Results from full mechanism
            key_species (list): List of species to consider
            
        Returns:
            float: Weighted species fitness metric
        """
        try:
            time = full_profiles["time"]
            fitness = 0.0
            raw_fitness = 0.0

            for species in key_species:
                if species not in reduced_profiles["mole_fractions"]:
                    print(f"Warning: Species '{species}' missing in reduced mechanism. Skipping.")
                    continue

                # Get weight for this species (default 0.0 if not specified)
                species_weight = self.weights["species"].get(species, 0.0)
                xi_red = reduced_profiles["mole_fractions"][species]
                xi_full = full_profiles[species]
                
                # Accumulate weighted errors
                raw_fitness += self.calculate_error(time, xi_red, xi_full)
                fitness += raw_fitness * species_weight
            
            return fitness
        except Exception as e:
            print(f"Error in species_fitness: {e}")
            return float("inf")

    def ignition_delay_fitness(self, reduced_profiles, full_profiles):
        """
        Calculate fitness based on ignition delay time agreement.
        
        Args:
            reduced_profiles (dict): Results from reduced mechanism
            full_profiles (dict): Results from full mechanism
            
        Returns:
            float: Ignition delay fitness metric
        """
        try:
            reduced_delay = reduced_profiles.get("ignition_delay", 0.0)
            full_delay = full_profiles.get("ignition_delay", 0.0)
            
            if self.difference_function == "absolute":
                raw_fitness = np.sqrt(((reduced_delay - full_delay) / full_delay)**2)
            elif self.difference_function == "squared":
                raw_fitness = ((reduced_delay - full_delay) / full_delay)**2
                
            return raw_fitness
        except Exception as e:
            print(f"Error in ignition_delay_fitness: {e}")
            return float("inf")
    
    def reaction_count_fitness(self, reduced_reactions, full_reactions):
        """
        Calculate fitness based on reaction count reduction.
        
        Args:
            reduced_reactions (int): Number of reactions in reduced mechanism
            full_reactions (int): Number of reactions in full mechanism
            
        Returns:
            float: Reaction count fitness metric
        """
        try:
            if self.difference_function == "absolute":
                raw_fitness = np.sqrt((reduced_reactions / full_reactions)**2)
            elif self.difference_function == "squared":
                raw_fitness = ((reduced_reactions) / full_reactions)**2
            else:
                raise ValueError(f"Unsupported difference function: {self.difference_function}")

            return raw_fitness
        except Exception as e:
            print(f"Error in reaction_count_fitness: {e}")
            return float("inf")
        
    def combined_fitness(self, reduced_profiles, full_profiles, key_species, reduced_reactions, genome_length):
        """
        Calculate comprehensive fitness combining all components with weights.
        
        Args:
            reduced_profiles (dict): Results from reduced mechanism
            full_profiles (dict): Results from full mechanism
            key_species (list): Important species to consider
            reduced_reactions (int): Number of reactions in reduced mechanism
            genome_length (int): Number of reactions in full mechanism
            
        Returns:
            dict: Dictionary containing all fitness components and combined score
        """
        try:
            # Calculate individual fitness components
            temp_fitness_raw = self.temperature_fitness(reduced_profiles, full_profiles) 
            species_fitness_raw = self.species_fitness(reduced_profiles, full_profiles, key_species)
            ignition_fitness_raw = self.ignition_delay_fitness(reduced_profiles, full_profiles)             
            reaction_fitness_raw = self.reaction_count_fitness(reduced_reactions, genome_length) 
            
            # Normalize components (except reaction count)
            temp_fitness_norm = self.normalize(temp_fitness_raw)
            species_fitness_norm = self.normalize(species_fitness_raw)
            ignition_fitness_norm = self.normalize(ignition_fitness_raw)
            
            # Apply weights to components
            temp_fitness = temp_fitness_norm * self.weights["temperature"]
            species_fitness = species_fitness_norm
            ignition_fitness = ignition_fitness_norm * self.weights["IDT"]
            reaction_fitness = reaction_fitness_raw * self.weights["reactions"] 
                
            # Calculate combined fitness
            combined_fitness = ((temp_fitness + species_fitness + ignition_fitness + reaction_fitness))
            
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