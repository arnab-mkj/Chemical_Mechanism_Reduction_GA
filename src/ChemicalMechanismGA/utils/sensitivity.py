
import sys
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import timeit

def ign_uq(g, factor, temp, pres, fuel, phi, T_equi, simtype, bootplot=True):
    # This function calculates the ignition delay time (IDT) 
    # for a given set of reaction rate multipliers
    # Calculates the ignition delay time (IDT), which is the time at which 
    # the system reaches the point of maximum temperature gradient (rapid heat release).
    
	gas.set_multiplier(1.0) # reset all multipliers
	for i in range(gas.n_reactions):
		gas.set_multiplier(factor[i],i)
		#print(gas.reaction_equations(i)+' index_reaction:',i,'multi_factor:',factor[i])
	
	gas.set_equivalence_ratio( phi, fuel, 'O2:1.0, N2:3.76' )
	gas.TP = temp, pres*ct.one_atm

	# here it is constant pressure with IdealGasReactor
	r = ct.IdealGasConstPressureReactor(gas)
	sim = ct.ReactorNet([r])
	
	# set the tolerances for the solution and for the sensitivity coefficients
	sim.rtol = 1.0e-6
	sim.atol = 1.0e-15

	t_end = 10
	time = []
	temp = []
	states = ct.SolutionArray(gas, extra=['t'])

	# stateArray = []
	while sim.time < t_end and r.T < T_equi - 0.1:
		sim.step()
		time.append(sim.time)
		temp.append(r.T)
		states.append(r.thermo.state, t=sim.time)
		# stateArray.append(r.thermo.X)

	time = np.array(time)
	temp = np.array(temp)
	diff_temp = np.diff(temp)/np.diff(time)

	if bootplot:
		# print(r.T, T_equi)
		plt.figure()
		plt.plot(states.t, states.T, '-ok')
		# plt.plot( 1.0, states.T[ign_pos], 'rs',markersize=6  )
		plt.xlabel('Time [ms]')
		plt.ylabel('T [K]')
		#plt.xlim( 0.99, 1.01 )
		#plt.ylim( -0.001,0.001 )
		plt.tight_layout()
		plt.show()

	ign_pos = np.argmax( diff_temp )
	ign = time[ign_pos]
	gas.set_multiplier(1.0) # reset all multipliers

	return ign #Returns the ignition delay time (IDT) as a single value



def ign_sens(g, temp, pres, fuel, phi, ign0, simtype, bootplot=True):
    # Calculates the sensitivity coefficients of IDT to the reaction rate constants (ki)
    # This tells us how much the IDT changes when the rate constant of reaction i is perturbed
    
    gas.set_multiplier(1.0)  # Reset all multipliers

    gas.set_equivalence_ratio(phi, fuel, 'O2:1.0, N2:3.76')
    gas.TP = temp, pres * ct.one_atm
    
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

    t_end = 10
    time = []
    temp = []

    # Run the simulation
    while sim.time < ign0 *1.5:
        sim.step()
        sens_T.append(sim.sensitivities()[2, :])  # Sensitivity of temperature
        states.append(r.thermo.state, t=sim.time)
        time.append(sim.time)
        temp.append(r.T)

    # Convert lists to numpy arrays
    time = np.array(time)
    temp = np.array(temp)
    diff_temp = np.diff(temp) / np.diff(time)  # Temperature gradient
    sens_T = np.array(sens_T)
    print(sens_T)

    # Find the ignition point
    ign_pos = np.argmax(diff_temp)
    print(ign_pos)
    ign = time[ign_pos]  #timeof the igniton point
    print(ign)

    # Optional: Plot temperature evolution
    if bootplot:
        plt.figure()
        plt.plot(states.t , states.T, '-ok')
        plt.plot(ign, states.T[ign_pos], 'rs', markersize=6)
        plt.xlabel('Time [ms]')
        plt.ylabel('T [K]')
        # plt.xlim(0.99, 1.01)
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

    # # 1. Sensitivity coefficients over time for selected reactions
    # selected_reactions = [0, 1, 2]  # Example: Plot for the first 3 reactions
    # plt.figure(figsize=(10, 6))
    # for i in selected_reactions:
    #     plt.plot(time, sens_T[:, i], label=f'Reaction {i}')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Sensitivity Coefficient (dT/dk)')
    # plt.title('Sensitivity of Temperature to Reaction Rate Constants Over Time')
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    # 2. Sensitivity coefficients at the ignition point (bar plot)
    reaction_indices = np.arange(len(sens_T_ign))  # Reaction indices
    plt.figure(figsize=(12, 6))
    plt.bar(reaction_indices, sens_T_ign, color='b', alpha=0.7)
    plt.xlabel('Reaction Index')
    plt.ylabel('Sensitivity Coefficient (dT/dk)')
    plt.title('Sensitivity of Temperature to Reaction Rate Constants at Ignition')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # # 3. Top N reactions influencing temperature at ignition
    # sorted_indices = np.argsort(-np.abs(sens_T_ign))  # Sort by absolute sensitivity
    # top_n = 5  # Number of top reactions to plot
    # top_reactions = sorted_indices[:top_n]
    # plt.figure(figsize=(12, 6))
    # plt.bar(top_reactions, sens_T_ign[top_reactions], color='r', alpha=0.7)
    # plt.xlabel('Reaction Index')
    # plt.ylabel('Sensitivity Coefficient (dT/dk)')
    # plt.title(f'Top {top_n} Reactions Influencing Temperature at Ignition')
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.show()

    return sens_T_ign, reaction_data #Returns a 1D array of sensitivity coefficients (S IDT) for all reactions in the mechanism.

def species_sens(gas, temp, pres, fuel, phi, species_list, simtype, ign0, bootplot=True):
    """
    Calculate the sensitivity of species concentrations to reaction rate constants.

    Args:
        gas (ct.Solution): Cantera gas object.
        temp (float): Initial temperature (K).
        pres (float): Initial pressure (atm).
        fuel (str): Fuel species (e.g., 'CH4').
        phi (float): Equivalence ratio.
        species_list (list): List of species to analyze (e.g., ['OH', 'CO']).
        simtype (str): Reactor type (e.g., 'HP' for constant pressure).
        ign0 (float): Ignition delay time (s).
        bootplot (bool): Whether to plot species concentrations.

    Returns:
        dict: Sensitivity coefficients for each species and reaction.
    """
    # Reset all multipliers
    gas.set_multiplier(1.0)

    # Set equivalence ratio and initial conditions
    gas.set_equivalence_ratio(phi, fuel, 'O2:1.0, N2:3.76')
    gas.TP = temp, pres * ct.one_atm

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

    # Run the simulation
    while sim.time < ign0 * 1.5:
        sim.step()
        time.append(sim.time)
        for species in species_list:
            species_concentrations[species].append(r.thermo[species].X)  # Mole fraction
            species_sensitivities[species].append(sim.sensitivities()[gas.species_index(species), :])

    # Convert lists to numpy arrays
    for species in species_list:
        species_concentrations[species] = np.array(species_concentrations[species])
        species_sensitivities[species] = np.array(species_sensitivities[species])

    # Optional: Plot species concentrations
    if bootplot:
        plt.figure(figsize=(10, 6))
        for species in species_list:
            plt.plot(time, species_concentrations[species], label=f"{species} Concentration")
        plt.xlabel("Time [s]")
        plt.ylabel("Mole Fraction")
        plt.title("Species Concentrations Over Time")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Extract sensitivities at the ignition point
    ign_pos = np.argmax(np.diff(np.array(time)))  # Approximate ignition point
    species_sens_at_ign = {species: species_sensitivities[species][ign_pos, :] for species in species_list}

    return species_sens_at_ign

pres = 2670 / 101325
temp = 1800.0
phi = .5
simtype = 'HP'

fuel = 'CH4'
mech = 'gri30.yaml'
dk = 5.e-2

# mech = 'h2_li_19.xml'
# fuel = 'nc7h16'
# mech = 'nc7sk88.cti'
# gas = ct.Solution('mech/dme_sk39.cti')
# gas = ct.Solution('mech/ic8sk143.cti')
# mech = "c4_49.xml"

string_list = [fuel, mech, str(phi), str(pres), str(temp), str(dk), simtype]
string_list = '_'.join(string_list)
print(string_list)

mechfile = 'gri30.yaml'
gas = ct.Solution(mechfile)
gas.set_equivalence_ratio( phi, fuel, 'O2:0.1938, N2:0.7287' )
gas.TP = temp, pres*ct.one_atm

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

# Create an empty column to store the sensitivities data. 
# baseCase for brute force method
ds["index"] = ""
ds["bruteforce"] = ""
ds["adjoint"] = ""
ds["ratio"] = ""

# base case
factor = np.ones( gas.n_reactions )
ign0 = ign_uq(gas, factor, temp, pres, fuel, phi, T_equi, simtype, True)
print("Ignition Delay is: {:.4f} ms".format(ign0*1000))

start_time = timeit.default_timer()
print('Start Adjoint')

#sens_T_ign, reaction_data = ign_sens(gas, temp, pres, fuel, phi, ign0, simtype, True)

# Save to a CSV file
# df = pd.DataFrame(reaction_data)
# df.to_csv("sensitivity_results_IDT.csv", index=False)
# print("Sensitivity results saved to sensitivity_results.csv")

elapsed = timeit.default_timer() - start_time
print('Finish Adjoint {:.2f}'.format(elapsed))

key_species = ['CH4','O2','CO2','H2O','CO','H2','O','OH','H','CH3']
species_sensitivities = species_sens(gas, temp, pres, fuel, phi, key_species, simtype, ign0, bootplot=True)

# Plot sensitivities for a specific species (e.g., OH)
species = 'OH'
reaction_indices = np.arange(len(species_sensitivities[species][0]))
plt.figure(figsize=(12, 6))
plt.bar(reaction_indices, species_sensitivities[species][0], color='b', alpha=0.7)
plt.xlabel('Reaction Index')
plt.ylabel(f'Sensitivity Coefficient (d[{species}]/dk)')
plt.title(f'Sensitivity of {species} Concentration to Reaction Rate Constants at Ignition')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

