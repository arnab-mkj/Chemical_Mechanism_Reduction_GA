import numpy as np
import cantera as ct
import pandas as pd
import json
import os

def ign_uq(gas, factor, temp, pres, fuel, phi, oxidizer_comp, T_equi, simtype):
    """
    Calculate the ignition delay time (IDT) for a given set of reaction rate multipliers.
    
    Args:
        gas: Cantera Solution object.
        factor: Array of reaction rate multipliers.
        temp: Initial temperature (K).
        pres: Pressure (Pa).
        fuel: Fuel species.
        phi: Equivalence ratio.
        oxidizer_comp: Oxidizer composition (e.g., 'O2:0.21, N2:0.79').
        T_equi: Equilibrium temperature (K).
        simtype: Simulation type (e.g., 'HP').
    
    Returns:
        tuple: Ignition delay time (s), time at equilibrium temperature (s).
    """
    gas.set_equivalence_ratio(phi, fuel, oxidizer_comp)
    gas.TP = temp, pres
    
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    
    sim.rtol = 1.0e-6
    sim.atol = 1.0e-15

    t_end = 0.1
    time = []
    temp = []
    states = ct.SolutionArray(gas, extra=['t'])

    while sim.time < t_end and r.T < T_equi:
        sim.step()
        time.append(sim.time)
        temp.append(r.T)
        states.append(r.thermo.state, t=sim.time)
    
    time = np.array(time)
    temp = np.array(temp)
    diff_temp = np.gradient(temp, time)
    ign_pos = np.argmax(diff_temp)
    ign = time[ign_pos]
    
    tequi_pos = np.argmin(np.abs(temp - T_equi))
    time_at_tequi = time[tequi_pos]
    
    return ign, time_at_tequi

def ign_sens(gas, temp, pres, fuel, phi, ign0, simtype):
    """
    Calculate sensitivity coefficients of IDT to reaction rate constants.
    
    Args:
        gas: Cantera Solution object.
        temp: Initial temperature (K).
        pres: Pressure (Pa).
        fuel: Fuel species.
        phi: Equivalence ratio.
        ign0: Baseline ignition delay time (s).
        simtype: Simulation type (e.g., 'HP').
    
    Returns:
        tuple: Sensitivity coefficients at ignition, reaction data dictionary.
    """
    gas.set_multiplier(1.0)
    gas.set_equivalence_ratio(phi, fuel, 'O2:0.21, N2:0.79')
    gas.TP = temp, pres
    
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    for i in range(gas.n_reactions):
        r.add_sensitivity_reaction(i)
    
    sim.rtol = 1.0e-6
    sim.atol = 1.0e-15
    sim.rtol_sensitivity = 1.0e-6
    sim.atol_sensitivity = 1.0e-6

    sens_T = []
    states = ct.SolutionArray(gas, extra=['t'])
    time = []
    temp = []

    t_end = 0.1
    while sim.time < t_end:
        sim.step()
        sens_T.append(sim.sensitivities()[2, :])
        states.append(r.thermo.state, t=sim.time)
        time.append(sim.time)
        temp.append(r.T)
    
    time = np.array(time)
    temp = np.array(temp)
    diff_temp = np.diff(temp) / np.diff(time)
    sens_T = np.array(sens_T)
    
    ign_pos = np.argmax(diff_temp)
    ign = time[ign_pos]
    sens_T_ign = sens_T[ign_pos, :]
    
    reaction_data = {
        "reaction_index": list(range(gas.n_reactions)),
        "reaction_equation": gas.reaction_equations(),
        "sensitivity_coefficient": list(sens_T_ign)
    }
    
    return sens_T_ign, reaction_data

def species_sens(gas, temp, pres, fuel, phi, oxidizer, species_list, simtype, ign0, time_equi):
    """
    Calculate the sensitivity of species concentrations to reaction rate constants.
    
    Args:
        gas: Cantera Solution object.
        temp: Initial temperature (K).
        pres: Pressure (Pa).
        fuel: Fuel species.
        phi: Equivalence ratio.
        species_list: List of species to analyze.
        simtype: Simulation type (e.g., 'HP').
        ign0: Baseline ignition delay time (s).
        time_equi: Time at equilibrium temperature (s).
    
    Returns:
        tuple: Species sensitivities, aggregate sensitivities, time array, sensitivities at ignition, aggregate sensitivities at ignition.
    """
    gas.set_multiplier(1.0)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = temp, pres
    
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    
    for i in range(gas.n_reactions):
        r.add_sensitivity_reaction(i)
    
    sim.rtol_sensitivity = 1.0e-6
    sim.atol_sensitivity = 1.0e-6

    species_sensitivities = {species: [] for species in species_list}
    time = []
    species_concentrations = {species: [] for species in species_list}
    aggregate_sensitivities = {species: [] for species in species_list}

    while sim.time < time_equi * 1.1:
        sim.step()
        time.append(sim.time)
        for species in species_list:
            species_concentrations[species].append(r.thermo[species].X)
            sensitivities = sim.sensitivities()[gas.species_index(species), :]
            species_sensitivities[species].append(sensitivities)
            aggregate_sensitivities[species].append(np.sum(np.abs(sensitivities)))
    
    for species in species_list:
        species_concentrations[species] = np.array(species_concentrations[species])
        species_sensitivities[species] = np.array(species_sensitivities[species])
        aggregate_sensitivities[species] = np.array(aggregate_sensitivities[species])
    
    time_ms = np.array(time) * 1000
    ign_pos = np.argmax(np.diff(np.array(time)))
    species_sens_at_ign = {species: species_sensitivities[species][ign_pos, :] for species in species_list}
    aggregate_sens_at_ign = {species: aggregate_sensitivities[species][ign_pos] for species in species_list}
    
    return species_sensitivities, aggregate_sensitivities, time, species_sens_at_ign, aggregate_sens_at_ign

def calculate_weights_at_ignition(aggregate_sens_at_ign, output_file):
    """
    Normalize the aggregated sensitivity values at the ignition point for all species
    and assign weights in the range [0, 1].
    
    Args:
        aggregate_sens_at_ign: Dictionary of aggregated sensitivities at ignition.
        output_file: Path to save the weights as a JSON file.
    
    Returns:
        dict: Normalized weights for each species.
    """
    species = list(aggregate_sens_at_ign.keys())
    sens_values = np.array([aggregate_sens_at_ign[sp] for sp in species])

    min_sens = np.min(sens_values)
    max_sens = np.max(sens_values)
    if max_sens == min_sens:
        normalized_weights = np.ones(len(species)) / len(species)
    else:
        normalized_weights = (sens_values - min_sens) / (max_sens - min_sens)

    weights = {species[i]: normalized_weights[i] for i in range(len(species))}
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        final_weights = {species: 1.0 / len(species) for species in species}
    else:
        final_weights = {species: weight / total_weight for species, weight in weights.items()}

    with open(output_file, 'w') as f:
        json.dump(final_weights, f, indent=4)
    
    return final_weights

def compute_species_weights(mechanism_path, conditions, key_species, output_file):
    """
    Compute species weights using sensitivity analysis for the given mechanism and conditions.
    
    Args:
        mechanism_path: Path to the chemical mechanism file (e.g., YAML).
        conditions: Dictionary with temperature, pressure, equivalence_ratio, fuel, oxidizer.
        key_species: List of species to compute weights for.
        output_file: Path to save the weights as a JSON file.
    
    Returns:
        dict: Normalized weights for each species.
    """
    gas = ct.Solution(mechanism_path)
    
    # Validate and normalize conditions
    required_keys = ['temperature', 'pressure', 'phi', 'fuel', 'oxidizer']
    if not all(key in conditions for key in required_keys):
        raise ValueError(f"Conditions dictionary must contain here: {required_keys}")
    
    phi = float(conditions.get('phi', 1.0))
    pres = float(conditions.get('pressure', 100000.0))
    temp = float(conditions.get('temperature', 1800.0))
    
    # Handle fuel and oxidizer formats
    fuel = conditions.get('fuel')
    if isinstance(fuel, dict):
        fuel = list(fuel.keys())[0]  # Use first fuel species if dict
    elif not isinstance(fuel, str):
        raise ValueError("Fuel must be a string or dictionary")
    
    oxidizer = conditions.get('oxidizer')
    if isinstance(oxidizer, dict):
        oxidizer_str = ", ".join(f"{k}:{v}" for k, v in oxidizer.items())
    elif isinstance(oxidizer, str):
        oxidizer_str = oxidizer
    else:
        raise ValueError("Oxidizer must be a string or dictionary")
    
    gas.set_equivalence_ratio(phi, fuel, oxidizer_str)
    gas.TP = temp, pres
    
    gas.equilibrate('HP')
    T_equi = gas.T
    
    factor = np.ones(gas.n_reactions)
    ign0, time_equi = ign_uq(
        gas,
        factor,
        temp,
        pres,
        fuel,
        phi,
        oxidizer_str,
        T_equi,
        'HP'
    )
    
    sens_T_ign, reaction_data = ign_sens(
        gas,
        temp,
        pres,
        fuel,
        phi,
        ign0,
        'HP'
    )
    
    df = pd.DataFrame(reaction_data)
    df.to_csv(os.path.join(os.path.dirname(output_file), "sensitivity_results_IDT.csv"), index=False)
    
    species_sensitivities, aggregate_sensitivities, time, species_sens_at_ign, aggregate_sens_at_ign = species_sens(
        gas,
        temp,
        pres,
        fuel,
        phi,
        oxidizer_str,
        key_species,
        'HP',
        ign0,
        time_equi
    )
    
    return calculate_weights_at_ignition(aggregate_sens_at_ign, output_file)