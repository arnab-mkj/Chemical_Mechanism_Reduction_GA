import sys
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import timeit
import json

def ign_uq(gas, factor, temp, pres, fuel, phi, oxidizer_comp, T_equi, simtype, bootplot=True):
    # This function calculates the ignition delay time (IDT) 
    # for a given set of reaction rate multipliers
    # Calculates the ignition delay time (IDT), which is the time at which 
    # the system reaches the point of maximum temperature gradient (rapid heat release).
    
    # gas.set_multiplier(1.0) # reset all multipliers
    # for i in range(gas.n_reactions):
    #     gas.set_multiplier(factor[i],i)
    #     #print(gas.reaction_equations(i)+' index_reaction:',i,'multi_factor:',factor[i])
    
    gas.set_equivalence_ratio( phi, fuel, oxidizer_comp )
    gas.TP = temp, pres
    
    # here it is constant pressure with IdealGasReactor
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
        
    # set the tolerances for the solution and for the sensitivity coefficients
    sim.rtol = 1.0e-6
    sim.atol = 1.0e-15

    t_end = 0.1 # in seconds 50ms
    time = []
    temp = []
    states = ct.SolutionArray(gas, extra=['t'])

    # stateArray = []
    while sim.time < t_end and r.T < T_equi:
        sim.step()
        time.append(sim.time)
        temp.append(r.T)
        states.append(r.thermo.state, t=sim.time)
        # stateArray.append(r.thermo.X)
    
    print("T_equi: ", T_equi)
    time = np.array(time)
    temp = np.array(temp)
    diff_temp = np.gradient(temp, time)
    ign_pos = np.argmax( diff_temp )
    ign = time[ign_pos]
    
    tequi_pos = np.argmin(np.abs(temp - T_equi))
    time_at_tequi = time[tequi_pos]
    
    print("Time at T_equi temp: ", time_at_tequi*1000, "ms")
    if bootplot:
        # Plot the temperature profile
        plt.figure(figsize=(8, 6))
        plt.plot(states.t * 1000, states.T, '-k', label="Temperature Profile")  # Convert time to ms
        plt.axvline(x=ign * 1000, color='r', linestyle='--', label="Ignition Delay (IDT)")  # Mark IDT point
        plt.axvline(x=time_at_tequi*1000, color='b', linestyle='--', label=f"Time at $T_{{equi}}$ ")
        plt.axhline(y=T_equi, color='g', linestyle='--', label=f"$T_{{equi}}$ = {T_equi:2f} K")
        # Add labels, title, and legend
        plt.xlabel('Time [ms]', fontsize=12)
        plt.ylabel('Temperature [K]', fontsize=12)
        plt.title('Temperature Profile with Ignition Delay', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # Adjust layout for better appearance
        plt.tight_layout()
        plt.show()

    
    # gas.set_multiplier(1.0) # reset all multipliers

    return ign, time_at_tequi #Returns the ignition delay time (IDT) as a single value


def ign_sens(gas, temp, pres, fuel, phi, ign0, simtype, bootplot=True):
    # Calculates the sensitivity coefficients of IDT to the reaction rate constants (ki)
    # This tells us how much the IDT changes when the rate constant of reaction i is perturbed
    
    gas.set_multiplier(1.0)  # Reset all multipliers

    gas.set_equivalence_ratio(phi, fuel, 'O2:2.0, N2:7.52')
    gas.TP = temp, pres
    
    r = ct.IdealGasConstPressureReactor(gas)

    sim = ct.ReactorNet([r])
    for i in range(gas.n_reactions):
        r.add_sensitivity_reaction(i)

    # Set tolerances for the simulation and sensitivity coefficients
    sim.rtol = 1.0e-6
    sim.atol = 1.0e-15
    sim.rtol_sensitivity = 1.0e-6
    sim.atol_sensitivity = 1.0e-6

    sens_T = []  # Store sensitivity coefficients of temperature
    states = ct.SolutionArray(gas, extra=['t'])  # Store thermodynamic states

    t_end = 0.1
    time = []
    temp = []

    # Run the simulation
    while sim.time < t_end:
        sim.step()
        sens_T.append(sim.sensitivities()[2, :])  # Sensitivity coeff of temperature with rate change
        states.append(r.thermo.state, t=sim.time)
        time.append(sim.time)
        temp.append(r.T)

    # Convert lists to numpy arrays
    time = np.array(time)
    temp = np.array(temp)
    diff_temp = np.diff(temp) / np.diff(time)  # Temperature gradient
    sens_T = np.array(sens_T)
    # print(sens_T)

    # Find the ignition point
    ign_pos = np.argmax(diff_temp)
    # print(ign_pos)
    ign = time[ign_pos]  #timeof the igniton point
    print("Sensitive ignition time: ", ign *1000, "ms")

    # Optional: Plot temperature evolution
    if bootplot:
        plt.figure()
        plt.plot(states.t*1000 , states.T, '-k')
        plt.plot(ign *1000, states.T[ign_pos], 'rs', markersize=6)
        plt.axvline(x=ign *1000, color='b', linestyle='--', label=f"Ignition_time:{ign*1000} ")
        plt.xlabel('Time [ms]')
        plt.ylabel('T [K]')
        plt.title('Sensitive Temperature Profile with Ignition Delay', fontsize=14)
        plt.xlim(0.00, 0.5) # in ms
        plt.legend(fontsize=10)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    
    # Sensitivity coefficients at the ignition point
    sens_T_ign = sens_T[ign_pos, :]
    
    # Create a dictionary to store results
    reaction_data = {
        "reaction_index": list(range(gas.n_reactions)),
        "reaction_equation": gas.reaction_equations(),
        "sensitivity_coefficient": list(sens_T_ign)
    }
    # --- Plotting Sensitivity Coefficients ---

    # 2. Sensitivity coefficients at the ignition point (bar plot)
    reaction_indices = np.arange(len(sens_T_ign))  # Reaction indices
    plt.figure(figsize=(12, 6))
    plt.bar(reaction_indices, sens_T_ign, color='b', alpha=0.7)
    plt.xlabel('Reaction Index', fontsize=10)
    plt.ylabel('Sensitivity Coefficient (dT/dk)', fontsize=10)
    plt.title('Sensitivity of Temperature to Reaction Rate Constants at Ignition')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # # 3. Top N reactions influencing temperature at ignition
    sorted_indices = np.argsort(-np.abs(sens_T_ign))  # Sort by absolute sensitivity
    top_n = 15  # Number of top reactions to plot
    top_reactions = sorted_indices[:top_n]
    # Get the reaction equations for the top reactions
    top_reaction_equations = [gas.reaction_equations()[i] for i in top_reactions]
    selected_reactions = []
    # Print the reaction numbers and equations
    print("Top Sensitive Reactions:")
    for idx, reaction_index in enumerate(top_reactions):
        selected_reactions.append(reaction_index+1)
        sensitivity_value = sens_T_ign[reaction_index]
        print(f"Rank {idx + 1}: Reaction {reaction_index + 1} -> {top_reaction_equations[idx]} \
              | Sensitivity: {sensitivity_value:.6f}")
    
    # Plot the top reactions influencing temperature at ignition
    plt.figure(figsize=(14, 7))
    ax = plt.gca()  # Get current axes
    # Plot horizontal bars
    for reaction_index, equation in zip(top_reactions, top_reaction_equations):
        ax.barh(
            f"R_{reaction_index}:{equation}",
            sens_T_ign[reaction_index],
            label=f"R_{reaction_index + 1}: {equation}",
            alpha=0.7,
            color='blue'
        )
    ax.set_ylabel('Reaction Index', fontsize=10)
    ax.set_xlabel('Sensitivity Coefficient (dT/dk)', fontsize=10)
    ax.set_title(f'Top {top_n} Reactions Influencing Temperature at Ignition')
    ax.grid(axis='x')
    # Move y-axis labels and ticks to the right
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    # Legend outside the plot on the right
    # ax.legend(title="Reactions", loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=6)
    plt.tight_layout()
    plt.show()
        
    
    # # 1. Sensitivity coefficients over time for selected reactions
      # Example: Plot for the first 3 reactions
    plt.figure(figsize=(10, 6))
    for i in selected_reactions:
        plt.plot(time*1000, sens_T[:, i], label=f'Reaction {i}')
    plt.xlabel('Time [ms]', fontsize=10)
    plt.ylabel('Sensitivity Coefficient (dT/dk)', fontsize=10)
    plt.title('Sensitivity of Temperature to Reaction Rate Constants Over Time')
    plt.legend()
    plt.grid()
    plt.xlim(0.00, 0.5) # in ms
    # plt.tight_layout()
    plt.show()
    
    return sens_T_ign, reaction_data #Returns a 1D array of sensitivity coefficients (S IDT) for all reactions in the mechanism.

def species_sens(gas, temp, pres, fuel, phi, species_list, simtype, ign0, time_equi, bootplot=True):
    """
    Calculate the sensitivity of species concentrations to reaction rate constants.

    Returns:
        dict: Sensitivity coefficients for each species and reaction.
    """
    # Reset all multipliers
    gas.set_multiplier(1.0)

    # Set equivalence ratio and initial conditions
    gas.set_equivalence_ratio(phi, fuel, 'O2:0.21, N2:0.79')
    gas.TP = temp, pres 

    # Set up the reactor
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])

    # Enable sensitivity analysis for all reactions
    for i in range(gas.n_reactions):
        r.add_sensitivity_reaction(i)

    # Set tolerances for sensitivity calculations
    sim.rtol_sensitivity = 1.0e-6
    sim.atol_sensitivity = 1.0e-6

    # Initialize arrays to store results
    species_sensitivities = {species: [] for species in species_list}
    time = []
    species_concentrations = {species: [] for species in species_list}
    aggregate_sensitivities = {species: [] for species in species_list}

    # Run the simulation
    while sim.time < time_equi * 1.1:
        sim.step()
        time.append(sim.time)
        for species in species_list:
            species_concentrations[species].append(r.thermo[species].X)  # Mole fraction
            sensitivities = sim.sensitivities()[gas.species_index(species), :]
            species_sensitivities[species].append(sensitivities)

            # Calculate aggregate sensitivity (sum of absolute sensitivities across all reactions)
            aggregate_sensitivities[species].append(np.sum(np.abs(sensitivities)))
   
    # Convert lists to numpy arrays
    for species in species_list:
        species_concentrations[species] = np.array(species_concentrations[species])
        species_sensitivities[species] = np.array(species_sensitivities[species])
        aggregate_sensitivities[species] = np.array(aggregate_sensitivities[species])
    
    # Normalize sensitivity vectors to get sensitivity directions
    sensitivity_directions = {}
    for species in species_list:
        norm = np.linalg.norm(species_sensitivities[species], axis=1, keepdims=True)
        sensitivity_directions[species] = species_sensitivities[species] / norm
    
    # print("Time (ms): ", np.array(time) * 1000)
    # for species in species_list:
    #     print(f"{species} Concentrations:", species_concentrations[species])
    #     print(f"{species} Aggregate Sensitivities:", aggregate_sensitivities[species])  
        
    time_ms = np.array(time) * 1000
    # Optional: Plot species concentrations
    if bootplot:
        plt.figure(figsize=(10, 6))
        for species in species_list:
            plt.plot(time_ms, species_concentrations[species], label=f"{species} Concentration")
        plt.axvline(x=ign0 * 1000, color='r', linestyle='--', label="Ignition Delay (IDT)")
        plt.xlabel("Time [ms]")
        plt.ylabel("Mole Fraction")
        plt.title("Species Concentrations Over Time")
        plt.xlim(0, max(time_ms))
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    # Plot aggregate sensitivities
    if bootplot:
        plt.figure(figsize=(10, 6))
        for species in species_list:
            plt.plot(time_ms, aggregate_sensitivities[species], label=f"{species} Aggregate Sensitivity")
        plt.axvline(x=ign0 * 1000, color='r', linestyle='--', label="Ignition Delay (IDT)")
        plt.xlabel("Time [ms]")
        plt.ylabel("Aggregate Sensitivity")
        plt.title("Aggregate Sensitivities of Species Over Time")
        plt.xlim(0, max(time_ms))
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        
    # Extract sensitivities at the ignition point
    ign_pos = np.argmax(np.diff(np.array(time)))  # Approximate ignition point
    species_sens_at_ign = {species: species_sensitivities[species][ign_pos, :] for species in species_list}
    aggregate_sens_at_ign = {species: aggregate_sensitivities[species][ign_pos] for species in species_list}
    
    #return sensitivity_directions, time, species_sens_at_ign, aggregate_sens_at_ign
    return species_sensitivities, aggregate_sensitivities, time, species_sens_at_ign, aggregate_sens_at_ign

def calculate_weights_at_ignition(aggregate_sens_at_ign, output_file):
    """
    Normalize the aggregated sensitivity values at the ignition point for all species
    and assign weights in the range [0, 1].

    Args:
        aggregate_sens_at_ign (dict): Aggregated sensitivities at the ignition point for each species.
        output_file (str): Path to save the weights as a JSON file.

    Returns:
        dict: Normalized weights for each species.
    """
    # Extract aggregated sensitivity values
    species = list(aggregate_sens_at_ign.keys())
    sens_values = np.array([aggregate_sens_at_ign[sp] for sp in species])

    # Normalize sensitivities to the range [0, 1]
    min_sens = np.min(sens_values)
    max_sens = np.max(sens_values)
    normalized_weights = (sens_values - min_sens) / (max_sens - min_sens)

    # Create a dictionary of weights
    weights = {species[i]: normalized_weights[i] for i in range(len(species))}
    
    # Step 3: Normalize the weights again to ensure they are in the range [0, 1]
    total_weight = sum(weights.values())

    # Normalize each weight by dividing by the total
    final_weights = {species: weight / total_weight for species, weight in weights.items()}

    # Save weights to a JSON file
    with open(output_file, 'w') as f:
        json.dump(final_weights, f, indent=4)

    print(f"Weights saved to {output_file}")
    return final_weights

config_path = "params.json"
with open(config_path, "r") as f:
        config = json.load(f)
        
key_species = config["key_species"]
condition = config["conditions"]
fuel = condition['fuel']
oxidizer_comp = condition['oxidizer']
phi = condition['equivalence_ratio']
pres = condition['pressure']
temp = condition['temperature']

simtype = 'HP'

mech = 'gri30.yaml'

gas = ct.Solution(mech)
gas.set_equivalence_ratio( phi, fuel, oxidizer_comp )
gas.TP = temp, pres

# print('species_names:')
# print(gas.species_names)
# print(gas.species_index('OH'))

# Get equilibrium temperature for ignition break
gas.equilibrate(simtype)
T_equi = gas.T

m = gas.n_reactions

# Create a dataframe to store sensitivity-analysis data
ds = pd.DataFrame(data=[], index=gas.reaction_equations(range(m)))
pd.options.display.float_format = '{:,.2e}'.format

# base case
factor = np.ones( gas.n_reactions ) #?????
ign0, time_equi = ign_uq(gas, factor, temp, pres, fuel, phi,oxidizer_comp, T_equi, simtype, True)
print("Ignition Delay is: {:.10f} ms".format(ign0*1000))

start_time = timeit.default_timer()
print('Start Adjoint')
sens_T_ign, reaction_data = ign_sens(gas, temp, pres, fuel, phi, ign0, simtype, True)
#Save to a CSV file
df = pd.DataFrame(reaction_data)
df.to_csv("sensitivity_results_IDT.csv", index=False)
print("Sensitivity results saved to sensitivity_results.csv")
elapsed = timeit.default_timer() - start_time
print('Finish Adjoint {:.2f}'.format(elapsed))


species_sensitivities, aggregate_sensitivities, time, species_sens_at_ign, aggregate_sens_at_ign = species_sens(
    gas, temp, pres, fuel, phi, key_species, simtype, ign0,time_equi, bootplot=True)
# Plot the evolution of sensitivity directions for a specific species (e.g., OH)

# Print aggregate sensitivities at ignition
print("Aggregate Sensitivities at Ignition:")
for species, agg_sens in aggregate_sens_at_ign.items():
    print(f"{species}: {agg_sens:.4e}")

output_file = "data/species_weights.json"
# Calculate and save weights
weights = calculate_weights_at_ignition(aggregate_sens_at_ign, output_file)

# Print the weights
print("Normalized Weights:")
for species, weight in weights.items():
    print(f"{species}: {weight:.4f}")
    
    
# species = 'CH4'
# top_reactions = [119, 158, 32, 156, 116, 155, 161, 121, 53, 85]
# plt.figure(figsize=(10, 6))
# for i in top_reactions:
#     plt.plot(np.array(time)/ign0, sensitivity_directions[species][:, i], label=f"Reaction {i}")
# plt.xlabel("Normalized Time [s]")
# plt.ylabel(f"Sensitivity Direction (OH)")
# plt.title(f"Evolution of Sensitivity Directions for {species}")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

