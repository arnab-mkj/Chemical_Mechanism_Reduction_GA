
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import random
import os

def create_reduced_mechanism(full_mech_file, reduction_fraction, reduced_mech_file):
    gas = ct.Solution(full_mech_file)
    n_reactions = gas.n_reactions
    
    non_duplicate_indices = [i for i in range(n_reactions) if not gas.reaction(i).duplicate]

    # Number of reactions to remove from non-duplicates
    n_remove = int(len(non_duplicate_indices) * reduction_fraction)
    print(f"Total reactions: {n_reactions}, Non-duplicates: {len(non_duplicate_indices)}, Removing: {n_remove}")

    # Randomly select reactions to remove from non-duplicates only
    remove_from_non_dup = sorted(random.sample(non_duplicate_indices, n_remove))

    # Keep all duplicates and non-duplicates except those removed
    keep_indices = [i for i in range(n_reactions) if (i not in remove_from_non_dup)]
    
    # Number of reactions to remove
    n_remove = int(n_reactions * reduction_fraction)
    print(f"Total reactions: {n_reactions}, Removing: {n_remove}")

    remove_indices = sorted(random.sample(range(n_reactions), n_remove)) # Randomly select reactions to remove
    keep_indices = [i for i in range(n_reactions) if i not in remove_indices]

    # Create a new gas object with only the kept reactions
    # we create a new mechanism string
    # We'll extract species and reactions, then write a new YAML with only kept reactions

    # Extract species definitions
    species = gas.species()
    # Extract reactions to keep
    reactions = [gas.reaction(i) for i in keep_indices]

    # Create new gas object from species and reactions
    reduced_gas = ct.Solution(thermo='ideal-gas', kinetics='GasKinetics',  transport="mixture-averaged",
                              species=species, reactions=reactions)

    # Save reduced mechanism to YAML file
    reduced_gas.write_yaml(reduced_mech_file)
    print(f"Reduced mechanism saved to {reduced_mech_file}")

    return reduced_mech_file

def run_simulation(mechanism, T0, P0, phi, species_list, t_end=0.05):
    gas = ct.Solution(mechanism)
    gas.TP = T, P
    air = "O2:0.21, N2:0.79" #mole fraction basis
    gas.set_equivalence_ratio(phi, fuel="CH4", oxidizer = air)

    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])

    time = 0.0
    time_step = 1e-6
    times = []
    temperatures = []
    time_points = np.arange(0, t_end + time_step, time_step)
    mole_fractions = {sp: [] for sp in species_list}

    for t in time_points:
        time = sim.advance(t)
        times.append(time)  # to ms
        temperatures.append(reactor.T)
        state = reactor.thermo.X
        for sp in species_list:
            idx = gas.species_index(sp)
            mole_fractions[sp].append(state[idx])

    return np.array(times), np.array(temperatures), mole_fractions

def calculate_error(time, reduced, full):
    reduced = np.array(reduced)
    full = np.array(full)
    diff_sq = (reduced - full) ** 2
    numerator = simpson(diff_sq )
    denominator = simpson(full ** 2)
    if denominator == 0:
        return float('inf')
    return numerator / denominator

if __name__ == "__main__":
    # Parameters
    full_mech = 'gri30.yaml'  # Full mechanism file path
    reduced_mech = 'reduced_test_gri30.yaml'  # Output reduced mechanism file path
    reduction_fraction = 0.3  # Remove 30% of reactions randomly

    T = 1800  # Initial temperature [K]
    P = ct.one_atm  # Pressure [Pa]
    phi = 1.0  # Equivalence ratio
    species_list = ['CH4', 'O2', 'CO2', 'H2O', 'CO', 'OH']

    # Create reduced mechanism
    
    create_reduced_mechanism(full_mech, reduction_fraction, reduced_mech)

    print(f"Reduced mechanism file {reduced_mech} already exists. Using existing file.")

    # Run simulations
    time_full, temp_full, mole_full = run_simulation(full_mech, T, P, phi, species_list)
    time_red, temp_red, mole_red = run_simulation(reduced_mech, T, P, phi, species_list)

    # Calculate errors
    errors = {}
    errors['temperature'] = calculate_error(time_full, temp_red, temp_full)
    for sp in species_list:
        errors[sp] = calculate_error(time_full, mole_red[sp], mole_full[sp])

    # Print errors
    print("Error in temperature profile:", errors['temperature'])
    for sp in species_list:
        print(f"Error in species {sp} mole fraction:", errors[sp])

    # Plot temperature profiles
    plt.figure(figsize=(10, 6))
    plt.plot(time_full*1000, temp_full, label='Full Mechanism')
    plt.plot(time_red*1000, temp_red, label='Reduced Mechanism', linestyle='--')
    plt.xlabel('Time [ms]')
    plt.ylabel('Temperature [K]')
    plt.title('Temperature Profile Comparison')
    plt.xlim(0.0, 0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot species mole fractions
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors
    for i, sp in enumerate(species_list):
        c = colors[i % len(colors)]
        plt.plot(time_full*1000, mole_full[sp], label=f'{sp} Full', color = c)
        plt.plot(time_red*1000, mole_red[sp], label=f'{sp} Reduced', linestyle='--', color =c)
    plt.xlabel('Time [ms]')
    plt.ylabel('Mole Fraction')
    plt.title('Species Mole Fractions Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0.0, 0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()