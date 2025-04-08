import numpy as np
from scipy.integrate import simpson
import math


class ConstantPressureError:
    def __init__(self, difference_function, sharpening_factor, lam):
       
        self.difference_function = difference_function
        self.sharpening_factor = sharpening_factor
        self.lam = lam

    def calculate_error(self, reduced, full, weight, time):
        
        numerator = simpson((reduced - full) ** 2)
        denominator = simpson(full ** 2)
        if denominator == 0:
            return float("inf")
        return weight * (numerator / denominator)

    def species_error(self, reduced_profiles, full_profiles, key_species, weights_sp, time):
        
        total_error = 0.0
        for species in key_species:
            if species in reduced_profiles and species in full_profiles:
                reduced = reduced_profiles[species]
                full = full_profiles[species]
                total_error += self.calculate_error(reduced, full, weights_sp, time)
        return total_error

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
        
        
    def temperature_error(self, reduced_temp, full_temp, weights_tmp, time):
        return self.calculate_error(reduced_temp, full_temp, weights_tmp, time)

   
    def combined_fitness(self, reduced_profiles, full_profiles, key_species, weights):
        time = full_profiles["time"]
        try:
            # Calculate individual errors
            species_err = self.species_error(reduced_profiles["mole_fractions"], full_profiles["mole_fractions"], key_species, weights["species"], time)
            temp_err = self.temperature_error(reduced_profiles["temperature_profile"],full_profiles["temperature_profile"], weights["temperature"], time)
            ignition_fitness = self.ignition_delay_fitness(reduced_profiles, full_profiles) * weights["ignition_delay"]
            # Total error
            total_error = species_err + temp_err + ignition_fitness

            # Fitness evaluation
            if total_error < self.lam: #lam taken as tol
                return total_error
            else:
                return float("inf")
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return float("inf")
        
        
        