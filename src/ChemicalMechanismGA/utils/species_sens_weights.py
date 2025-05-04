import numpy as np
import cantera as ct
import pandas as pd
import json
import os

def ign_uq(gas, factor, temp, pres, fuel, phi, oxidizer_comp, T_equi, simtype):
    """
    Calculate ignition delay time (IDT) and time to reach equilibrium temperature.
    
    Args:
        gas: Cantera Solution object representing the chemical mechanism
        factor: Array of reaction rate multipliers (not currently used)
        temp: Initial temperature in Kelvin
        pres: Pressure in Pascals
        fuel: Fuel species name (string)
        phi: Equivalence ratio (float)
        oxidizer_comp: Oxidizer composition string (e.g., 'O2:0.21,N2:0.79')
        T_equi: Equilibrium temperature in Kelvin (stopping criterion)
        simtype: Simulation type ('HP' for constant pressure)
    
    Returns:
        tuple: (ignition_delay_time, time_at_equilibrium_temperature) both in seconds
    """
    # Set initial gas mixture conditions
    gas.set_equivalence_ratio(phi, fuel, oxidizer_comp)
    gas.TP = temp, pres
    
    # Create reactor and simulation network
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    
    # Set numerical tolerances
    sim.rtol = 1.0e-6  # Relative tolerance
    sim.atol = 1.0e-15  # Absolute tolerance

    # Simulation parameters
    t_end = 0.1  # Maximum simulation time in seconds
    time = []
    temp = []
    states = ct.SolutionArray(gas, extra=['t'])  # For storing state history

    # Run time integration
    while sim.time < t_end and r.T < T_equi:
        sim.step()
        time.append(sim.time)
        temp.append(r.T)
        states.append(r.thermo.state, t=sim.time)
    
    # Calculate ignition delay as point of maximum temperature gradient
    time = np.array(time)
    temp = np.array(temp)
    diff_temp = np.gradient(temp, time)  # Temperature derivative
    ign_pos = np.argmax(diff_temp)  # Ignition point index
    ign = time[ign_pos]  # Ignition delay time
    
    # Find time when equilibrium temperature is reached
    tequi_pos = np.argmin(np.abs(temp - T_equi))
    time_at_tequi = time[tequi_pos]
    
    return ign, time_at_tequi

def ign_sens(gas, temp, pres, fuel, phi, ign0, simtype):
    """
    Compute sensitivity coefficients of ignition delay to reaction rates.
    
    Args:
        gas: Cantera Solution object
        temp: Initial temperature in Kelvin
        pres: Pressure in Pascals  
        fuel: Fuel species name
        phi: Equivalence ratio
        ign0: Baseline ignition delay time in seconds
        simtype: Simulation type ('HP' for constant pressure)
    
    Returns:
        tuple: (sensitivity_coefficients, reaction_data_dict)
    """
    # Reset reaction multipliers and set initial conditions
    gas.set_multiplier(1.0)
    gas.set_equivalence_ratio(phi, fuel, 'O2:0.21,N2:0.79')
    gas.TP = temp, pres
    
    # Set up reactor with sensitivity analysis
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    # Enable sensitivity for all reactions
    for i in range(gas.n_reactions):
        r.add_sensitivity_reaction(i)
    
    # Set numerical tolerances
    sim.rtol = 1.0e-6
    sim.atol = 1.0e-15 
    sim.rtol_sensitivity = 1.0e-6
    sim.atol_sensitivity = 1.0e-6

    # Initialize data storage
    sens_T = []  # Temperature sensitivities
    states = ct.SolutionArray(gas, extra=['t'])
    time = []
    temp = []

    # Run simulation
    t_end = 0.1
    while sim.time < t_end:
        sim.step()
        sens_T.append(sim.sensitivities()[2, :])  # Store temperature sensitivities
        states.append(r.thermo.state, t=sim.time)
        time.append(sim.time)
        temp.append(r.T)
    
    # Process results
    time = np.array(time)
    temp = np.array(temp)
    diff_temp = np.diff(temp) / np.diff(time)  # Temperature gradient
    sens_T = np.array(sens_T)
    
    # Find ignition point and corresponding sensitivities
    ign_pos = np.argmax(diff_temp)
    ign = time[ign_pos]
    sens_T_ign = sens_T[ign_pos, :]  # Sensitivities at ignition
    
    # Package reaction data
    reaction_data = {
        "reaction_index": list(range(gas.n_reactions)),
        "reaction_equation": gas.reaction_equations(),
        "sensitivity_coefficient": list(sens_T_ign)
    }
    
    return sens_T_ign, reaction_data

def species_sens(gas, temp, pres, fuel, phi, oxidizer, species_list, simtype, ign0, time_equi):
    """
    Compute species concentration sensitivities to reaction rates.
    
    Args:
        gas: Cantera Solution object
        temp: Initial temperature in Kelvin
        pres: Pressure in Pascals
        fuel: Fuel species name
        phi: Equivalence ratio
        oxidizer: Oxidizer composition string
        species_list: List of species to analyze
        simtype: Simulation type
        ign0: Baseline ignition delay time
        time_equi: Time to reach equilibrium temperature
    
    Returns:
        tuple: (species_sensitivities, aggregate_sensitivities, time_array,
                sensitivities_at_ignition, aggregate_at_ignition)
    """
    # Reset reaction multipliers and set initial conditions
    gas.set_multiplier(1.0)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = temp, pres
    
    # Set up reactor with sensitivity analysis
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    for i in range(gas.n_reactions):
        r.add_sensitivity_reaction(i)
    
    # Set sensitivity tolerances
    sim.rtol_sensitivity = 1.0e-6
    sim.atol_sensitivity = 1.0e-6

    # Initialize data storage
    species_sensitivities = {species: [] for species in species_list}
    time = []
    species_concentrations = {species: [] for species in species_list}
    aggregate_sensitivities = {species: [] for species in species_list}

    # Run simulation until slightly past equilibrium time
    while sim.time < time_equi * 1.1:
        sim.step()
        time.append(sim.time)
        for species in species_list:
            # Store species mole fractions and sensitivities
            species_concentrations[species].append(r.thermo[species].X)
            sensitivities = sim.sensitivities()[gas.species_index(species), :]
            species_sensitivities[species].append(sensitivities)
            # Calculate aggregate sensitivity (sum of absolute values)
            aggregate_sensitivities[species].append(np.sum(np.abs(sensitivities)))
    
    # Convert to numpy arrays
    for species in species_list:
        species_concentrations[species] = np.array(species_concentrations[species])
        species_sensitivities[species] = np.array(species_sensitivities[species])
        aggregate_sensitivities[species] = np.array(aggregate_sensitivities[species])
    
    # Find sensitivities at ignition point
    time_ms = np.array(time) * 1000  # Convert to milliseconds
    ign_pos = np.argmax(np.diff(np.array(time)))  # Approximate ignition point
    species_sens_at_ign = {
        species: species_sensitivities[species][ign_pos, :] 
        for species in species_list
    }
    aggregate_sens_at_ign = {
        species: aggregate_sensitivities[species][ign_pos] 
        for species in species_list
    }
    
    return (species_sensitivities, aggregate_sensitivities, time, 
            species_sens_at_ign, aggregate_sens_at_ign)

def calculate_weights_at_ignition(aggregate_sens_at_ign, output_file):
    """
    Normalize sensitivity values to create species weights.
    
    Args:
        aggregate_sens_at_ign: Dictionary of {species: aggregate_sensitivity}
        output_file: Path to save JSON file with weights
    
    Returns:
        dict: Normalized weights for each species
    """
    species = list(aggregate_sens_at_ign.keys())
    sens_values = np.array([aggregate_sens_at_ign[sp] for sp in species])

    # Normalize sensitivities to [0,1] range
    min_sens = np.min(sens_values)
    max_sens = np.max(sens_values)
    if max_sens == min_sens:  # Handle uniform sensitivities
        normalized_weights = np.ones(len(species)) / len(species)
    else:
        normalized_weights = (sens_values - min_sens) / (max_sens - min_sens)

    weights = {species[i]: normalized_weights[i] for i in range(len(species))}
    
    # Normalize to sum to 1
    total_weight = sum(weights.values())
    if total_weight == 0:  # Handle zero weights case
        final_weights = {species: 1.0 / len(species) for species in species}
    else:
        final_weights = {species: weight / total_weight for species, weight in weights.items()}

    # Save weights to JSON file
    with open(output_file, 'w') as f:
        json.dump(final_weights, f, indent=4)
    
    return final_weights

def compute_species_weights(mechanism_path, conditions, key_species, output_file):
    """
    Main workflow to compute species weights from sensitivity analysis.
    
    Args:
        mechanism_path: Path to mechanism file (YAML/CTI)
        conditions: Dictionary with simulation conditions
        key_species: List of species to analyze
        output_file: Path to save results
    
    Returns:
        dict: Normalized species weights
    """
    # Load chemical mechanism
    gas = ct.Solution(mechanism_path)
    
    # Validate input conditions
    required_keys = ['temperature', 'pressure', 'phi', 'fuel', 'oxidizer']
    if not all(key in conditions for key in required_keys):
        raise ValueError(f"Missing required condition: {required_keys}")
    
    # Extract and normalize conditions
    phi = float(conditions.get('phi', 1.0))
    pres = float(conditions.get('pressure', 100000.0))  # Default 1 bar
    temp = float(conditions.get('temperature', 1800.0))  # Default 1800K
    
    # Process fuel input (accept string or dict)
    fuel = conditions.get('fuel')
    if isinstance(fuel, dict):
        fuel = list(fuel.keys())[0]  # Use first fuel if dictionary
    elif not isinstance(fuel, str):
        raise ValueError("Fuel must be string or dictionary")
    
    # Process oxidizer input
    oxidizer = conditions.get('oxidizer')
    if isinstance(oxidizer, dict):
        oxidizer_str = ", ".join(f"{k}:{v}" for k, v in oxidizer.items())
    elif isinstance(oxidizer, str):
        oxidizer_str = oxidizer
    else:
        raise ValueError("Oxidizer must be string or dictionary")
    
    # Set initial state and calculate equilibrium
    gas.set_equivalence_ratio(phi, fuel, oxidizer_str)
    gas.TP = temp, pres
    gas.equilibrate('HP')
    T_equi = gas.T
    
    # Calculate baseline ignition characteristics
    factor = np.ones(gas.n_reactions)
    ign0, time_equi = ign_uq(
        gas, factor, temp, pres, fuel, phi, oxidizer_str, T_equi, 'HP'
    )
    
    # Calculate ignition delay sensitivities
    sens_T_ign, reaction_data = ign_sens(
        gas, temp, pres, fuel, phi, ign0, 'HP'
    )
    
    # Save sensitivity results
    df = pd.DataFrame(reaction_data)
    df.to_csv(os.path.join(os.path.dirname(output_file), "sensitivity_results_IDT.csv"), index=False)
    
    # Calculate species sensitivities
    species_results = species_sens(
        gas, temp, pres, fuel, phi, oxidizer_str, 
        key_species, 'HP', ign0, time_equi
    )
    
    # Calculate and return normalized weights
    return calculate_weights_at_ignition(species_results[3], output_file)