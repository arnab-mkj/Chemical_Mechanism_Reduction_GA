import numpy as np
import math
import matplotlib.pyplot as plt

# Full mechanism data (GRI-Mech 3.0)
n_total = 325
idt_full = 0.002  # seconds
temp_full = 2200  # K
species_full = {
    "CH4": 0.01,
    "O2": 0.15,
    "CO2": 0.040,
    "H2O": 0.12
}

# Define ranges for reduced mechanism values
temp_range = (800, 3000)  # K
idt_range = (0.0001, 0.01)  # s
n_active_range = (100, 325)
species_ranges = {
    "CH4": (0.001, 0.10),
    "O2": (0.001, 0.25),
    "CO2": (0.001, 0.10),
    "H2O": (0.001, 0.30)
}

# Parameter x from 0 to 1 (interpolates between min and max of each range)
x_values = np.linspace(0, 1, 200)

# Configurations
sharpening_factors = [0.5, 1, 5, 10]  # Different sigma values
diff_methods = ["linear", "squared"]
configs = [(sf, dm) for sf in sharpening_factors for dm in diff_methods]

# Collect normalized error data
results_data = {config: {"idt_error": [], "temp_error": [], "species_error": [], "n_active": []} 
                for config in configs}

# Store actual parameter values
parameter_values = {
    "temp": [],
    "idt": [],
    "n_active": [],
    "species": []
}

# Sigmoid normalization function
def sigmoid_normalize(raw_fitness, sharpening_factor):
    return 1 / (1 + math.exp(-sharpening_factor * raw_fitness))

def logarithmic_normalize(raw_fitness, sharpening_factor):
    return math.log(1 + sharpening_factor * abs(raw_fitness))

# Compute normalized errors for each x value
for x in x_values:
    # Generate reduced mechanism data
    temp_reduced = temp_range[0] + x * (temp_range[1] - temp_range[0])
    idt_reduced = idt_range[0] + x * (idt_range[1] - idt_range[0])
    n_active = int(n_active_range[0] + x * (n_active_range[1] - n_active_range[0]))
    species_reduced = {
        species: species_ranges[species][0] + x * (species_ranges[species][1] - species_ranges[species][0])
        for species in species_full
    }
    
    # Store actual parameter values
    parameter_values["temp"].append(temp_reduced)
    parameter_values["idt"].append(idt_reduced)
    parameter_values["n_active"].append(n_active)
    parameter_values["species"].append(np.mean(list(species_reduced.values())))
    
    # Compute errors for each configuration
    for sharpening_factor, diff_method in configs:
        # Compute raw errors
        if diff_method == "linear":
            error_temp = abs(temp_reduced - temp_full) / (temp_full)
            error_idt = abs(idt_reduced - idt_full) / (idt_full)
            
            # Calculate species error as mean absolute error
            species_errors = [abs(species_reduced[s] - species_full[s]) / 
                            (species_full[s]) 
                            for s in species_full]
            error_species = np.sum(species_errors)
            
            error_reaction = abs(n_active) / (n_total)
        else:  # Squared
            error_temp = ((temp_reduced - temp_full) / (temp_full)) ** 2
            error_idt = ((idt_reduced - idt_full) / (idt_full)) ** 2
            
            # Calculate species error as mean squared error
            species_errors = [((species_reduced[s] - species_full[s]) / 
                             (species_full[s])) ** 2
                             for s in species_full]
            error_species = np.sum(species_errors)
            
            error_reaction = ((n_active) / (n_total)) ** 2

        # Normalize errors using sigmoid function
        f_temp = sigmoid_normalize(error_temp, sharpening_factor)
        f_idt = sigmoid_normalize(error_idt, sharpening_factor)
        f_species = sigmoid_normalize(error_species, sharpening_factor)
        f_reaction = sigmoid_normalize(error_reaction, sharpening_factor)

        # Store normalized errors
        results_data[(sharpening_factor, diff_method)]["idt_error"].append(f_idt)
        results_data[(sharpening_factor, diff_method)]["temp_error"].append(f_temp)
        results_data[(sharpening_factor, diff_method)]["species_error"].append(f_species)
        results_data[(sharpening_factor, diff_method)]["n_active"].append(error_reaction)

# Plot results
plt.figure(figsize=(12, 8))

# a) Normalized Ignition Delay Error
plt.subplot(2, 2, 1)
for sharpening_factor, diff_method in configs:
    label = f"σ={sharpening_factor} ({diff_method.capitalize()})"
    plt.plot(parameter_values["idt"], results_data[(sharpening_factor, diff_method)]["idt_error"], 
             label=label)
plt.xlabel("Ignition Delay Time (s)")
plt.ylabel("Normalized Error")
plt.legend()
plt.grid()
plt.title("a) Normalized Ignition Delay Error")
plt.axvline(x=idt_full, color='gray', linestyle='--', label='Full Mechanism')

# b) Normalized Steady-State Temperature Error
plt.subplot(2, 2, 2)
for sharpening_factor, diff_method in configs:
    label = f"σ={sharpening_factor} ({diff_method.capitalize()})"
    plt.plot(parameter_values["temp"], results_data[(sharpening_factor, diff_method)]["temp_error"], 
             label=label)
plt.xlabel("Temperature (K)")
plt.ylabel("Normalized Error")
plt.legend()
plt.grid()
plt.title("b) Normalized Temperature Error")
plt.axvline(x=temp_full, color='gray', linestyle='--', label='Full Mechanism')

# c) Normalized Species Error (average across all species)
plt.subplot(2, 2, 3)
for sharpening_factor, diff_method in configs:
    label = f"σ={sharpening_factor} ({diff_method.capitalize()})"
    plt.plot(parameter_values["species"], results_data[(sharpening_factor, diff_method)]["species_error"], 
             label=label)
plt.xlabel("Average Species Concentration (mole fraction)")
plt.ylabel("Normalized Error")
plt.legend()
plt.grid()
plt.title("c) Normalized Species Error (average)")
plt.axvline(x=np.mean(list(species_full.values())), color='gray', linestyle='--', label='Full Mechanism')

# d) Normalized Reaction Count Error
plt.subplot(2, 2, 4)
for sharpening_factor, diff_method in configs:
    label = f"σ={sharpening_factor} ({diff_method.capitalize()})"
    plt.plot(parameter_values["n_active"], results_data[(sharpening_factor, diff_method)]["n_active"], 
             label=label)
plt.xlabel("Number of Active Reactions")
plt.ylabel("Normalized Error")
plt.legend()
plt.grid()
plt.title("d) Normalized Reaction Count Error")
plt.axvline(x=n_total, color='gray', linestyle='--', label='Full Mechanism')

plt.tight_layout()
plt.savefig("fitness_analysis_sigmoid_ranges_gri30_corrected.png")
plt.show()