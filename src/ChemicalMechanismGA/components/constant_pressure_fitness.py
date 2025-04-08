import numpy as np
import cantera as ct
import math


class ConstantPressureFitness:
    def __init__(self, difference_function, sharpening_factor, lam):
        self.difference_function = difference_function
        self.sharpening_factor = sharpening_factor
        self.lam = lam

    def calculate_difference_function(self, reduced, full):
        """Calculate the difference between actual and full values using the specified method."""
        try:
            if self.difference_function == "absolute":
                return abs((reduced - full)/full)
            
            elif self.difference_function == "squared":
                return ((reduced - full)/full) ** 2
            
            elif self.difference_function == "logarithmic":
                if full == 0:
                    raise ValueError("full value is zero, cannot calculate logarithmic difference.")
                return math.log(1 + self.sharpening_factor * abs((reduced - full) / full))
            
            elif self.difference_function == "sigmoid":
                if full == 0:
                    raise ValueError("full value is zero, cannot calculate sigmoid difference.")
                return 1 / (1 + math.exp(self.sharpening_factor * ( reduced / (self.lam*full) - 1)))
            
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
        
    def integral_profile(self, time, xi_red, xi_full):
        if len(time) != len(xi_red) or len(time) != len(xi_full):
            raise ValueError("Input arrays 'time', 'xi_red', and 'xi_full' must have the same length.")
        if len(time) < 2:
            raise ValueError("Input arrays must have at least two time points.")
                
        if self.difference_function == "sigmoid_integral":
            # Calculate the absolute difference between profiles
            abs_difference = np.abs(xi_red - xi_full)
            integral = np.trapz(abs_difference, time)
            # Calculate the normalization factor (A_ref)
            xi_full_max = np.max(xi_full)
            xi_full_min = np.min(xi_full)
            delta_time = time[-1] - time[0]
            # print(f"time length: {len(time)}, time: {time}")
            # print(f"delta time: {delta_time}")
            A_ref = ((xi_full_max - xi_full_min)) * delta_time
            if A_ref == 0:
                raise ValueError("A_ref is zero, normalization is not possible.")
            integral_profile = integral / A_ref       
            fitness = ((1 / (1 + math.exp((-1) * self.sharpening_factor * integral_profile)))) - 1
            # Return the normalized profile difference
            return fitness
        else:
            raise ValueError(f"Unsupported difference function: {self.difference_function}")


    def temperature_fitness(self, reduced_profiles, full_profiles):
        """Calculate fitness based on temperature profile."""
        try:
            if self.difference_function in {"linear", "squared", "logarithmic"}:
                temp_red = np.max(reduced_profiles["temperature_profile"])
                temp_full = np.max(full_profiles["temperature_profile"])
                return self.calculate_difference_function(temp_red, temp_full)
                
            elif self.difference_function == "sigmoid_integral":
                time = full_profiles["time"]
                temp_red = reduced_profiles["temperature_profile"]
                temp_full = full_profiles["temperature_profile"]
                # print(f"temp reduced profile: {temp_red}")
                # print(f"temp full profile: {temp_full}")
                
                return self.integral_profile(time, temp_red, temp_full)
        except Exception as e:
            print(f"Error in temperature_fitness: {e}")
            return float("inf")

    def species_fitness(self, reduced_profiles, full_profiles, key_species):
        """Calculate fitness based on species mole fractions."""
        try:
            time = full_profiles["time"]
            reduced_mole_fractions = reduced_profiles["mole_fractions"]
                
            fitness = 0.0
            valid_species_count = 0
            for species in key_species:
                if species not in reduced_mole_fractions:
                    print(f"Warning: Species '{species}' is missing in the reduced mechanism. Skipping.")
                    continue
                if species in reduced_profiles and species in full_profiles:
                    if self.difference_function in {"linear", "squared", "logarithmic"}:
                        xi_red = np.max(reduced_profiles[species])
                        xi_full = np.max(full_profiles[species])
                        return self.calculate_difference_function(xi_red, xi_full)
                        
                    
                    elif self.difference_function == "sigmoid_integral":
                        xi_red = reduced_profiles[species]
                        xi_full = full_profiles[species]
                        # print(f"species reduced profile: {xi_red}")
                        # print(f"species full profile: {xi_full}")
                        fitness += self.integral_profile(time, xi_red, xi_full)
                    valid_species_count += 1
                    
            return fitness / valid_species_count if valid_species_count > 0 else float("inf")
        except Exception as e:
            print(f"Error in species_fitness: {e}")
            return float("inf")

    def ignition_delay_fitness(self, reduced_profiles, full_profiles):
        """Calculate fitness based on ignition delay time."""
        try:
            if self.difference_function in {"linear", "squared", "logarithmic", "sigmoid_integral"}:
                reduced_delay = reduced_profiles.get("ignition_delay", 0.0)
                full_delay = full_profiles.get("ignition_delay", 0.0)
                return math.log(1 + self.sharpening_factor * abs((reduced_delay - full_delay) / full_delay))
        except Exception as e:
            print(f"Error in ignition_delay_fitness: {e}")
            return float("inf")

    def combined_fitness(self, reduced_profiles, full_profiles, key_species, weights):
        """Calculate combined fitness using weighted components."""
        try:
            temp_fitness = self.temperature_fitness(reduced_profiles, full_profiles) * weights["temperature"]
            # species_fitness = self.species_fitness(reduced_profiles, full_profiles, key_species) * weights["species"]
            ignition_fitness = self.ignition_delay_fitness(reduced_profiles, full_profiles) * weights["ignition_delay"]
            total_weight = sum(weights.values())
            return (temp_fitness  + ignition_fitness) / total_weight
            #return (temp_fitness + species_fitness) 
        except Exception as e:
            print(f"Error in combined_fitness: {e}")
            return float("inf")
