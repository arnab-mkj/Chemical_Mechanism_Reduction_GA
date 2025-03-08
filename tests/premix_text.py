import numpy as np
import cantera as ct
from scipy import integrate
"""
    This is to test the fitness functions of full mechanism along with 
    a reduced mechanism.
    Implemented Fitness:
        a) PREMIX: with -Freeflame(computes flame speed)
                        -Burnerflame(for a fixed mass flow rate)
        
"""

def calculate_premix_fitness(species_reduced, conditions_list, full_mech, reduced_mech, epsilon=1e-8):
    total_error = 0.0
    
    
    for j, condition in enumerate(conditions_list):
        print(f"Processing condition {j+1}/{len(conditions_list)}")
        
        # Run simulations for this condition
        full_profiles = run_premix_simulation(full_mech, condition)
        reduced_profiles = run_premix_simulation(reduced_mech, condition)
        
        z = full_profiles['grid']
        
        condition_error = 0.0
        for k, species in enumerate(species_reduced):
            
            if species not in full_profiles or species not in reduced_profiles:
                continue
            
            Y_orig = full_profiles[species]
            Y_calcd = reduced_profiles[species]
            
            W_k = 1.0 if np.max(Y_orig) >= 1e-7 else 0.0
            if W_k == 0.0:
                continue
            l2_orig = np.sqrt(integrate.trapz(Y_orig**2, z))
            
            diff = Y_calcd - Y_orig
            l2_diff = np.sqrt(integrate.trapz(diff**2, z))
            
            if l2_orig > 0:
                relative_error = W_k * l2_diff / l2_orig
            else:
                relative_error = 0 if l2_diff == 0 else W_k # handle edge cases
                
            condition_error += relative_error
        
        total_error += condition_error
            
    return 1.0/(epsilon + total_error)
            

def run_premix_simulation(mechanism_file, condition):
    gas = ct.Solution(mechanism_file)
    
    # Set initial state based on condition
    # fuel_comp = condition.get('fuel', {'CH4': 1.0})
    # oxidizer_comp = condition.get('oxidizer', {'O2': 0.21, 'N2': 0.79})
    # equivalence_ratio = condition.get('phi', 1.0)
    # pressure = condition.get('pressure', ct.one_atm)
    # temperature = condition.get('temperature', 300.0)
    # mass_flow_rate = condition.get('mass_flow_rate', 0.04) #kg/m^2/s
    
    # Extract values from condition
    fuel_comp = condition['fuel']
    oxidizer_comp = condition['oxidizer']
    equivalence_ratio = condition['phi']
    pressure = condition['pressure']
    temperature = condition['temperature']
    mass_flow_rate = condition['mass_flow_rate']
    
    mixture = {}
    for fuel, fuel_val in fuel_comp.items():
        mixture[fuel] = fuel_val
    for ox, ox_val in oxidizer_comp.items():
        mixture[ox] = ox_val
    # print("mixture:", mixture)
        
    gas.TPX = temperature, pressure, mixture
    # print(gas.X)
    
    mdot = condition['mass_flow_rate']
    # Burner-stabilized flame
    flame = ct.BurnerFlame(gas, width=0.05)
    flame.burner.mdot = mdot  # g/(cm²·s)

    flame.transport_model = 'mixture-averaged'
   
    #Set simulation parameters
    flame.set_refine_criteria(ratio=3.0, slope=0.3, curve=1)
    # flame.max_time_step_count = 900
    
    flame.energy_enabled = False
    try:
        # Solve the flame
        flame.solve(loglevel=1, refine_grid=True)
        print("flame solved")  #debugging line
        flame.save('flame_fixed_T.csv', basis="mole", overwrite=True)
        flame.show_stats()

        # Collect results
        results = {'grid': flame.grid}
        for sp in gas.species_names:
            # print(f"Working on result: {sp}")  #debugging line
            # print(f"Grid size: {len(flame.grid)}")
            # print(f"Shape of flame.X: {flame.X.shape}")
            results[sp] = flame.X[gas.species_index(sp),:]
        return results
    
    except Exception as e:
        print(f"Error in flame simulation: {e}")
        # Return empty profiles if simulation fails
        empty_array = np.array([0.0])
        return {
            "grid": empty_array,
            "temperature_profile": empty_array,
            "mole_fractions": np.zeros((len(gas.species_names), 1)),
            "error": str(e)
            }

def evaluate_reduction_mechanism(full_mech, reduced_mech, conditions_list):
    
    # full_profiles= run_premix_simulation(full_mech, conditions)
    # reduced_profiles = run_premix_simulation(reduced_mech, conditions)
    
    gas = ct.Solution(reduced_mech) #implement a class system later so that it is not repeated
    species_reduced = gas.species_names
    
    fitness = calculate_premix_fitness(species_reduced, conditions_list, full_mech, reduced_mech)
    return fitness


if __name__== "__main__":
    # conditions_list = [
    #     # Condition 1: Lean flame (phi=0.8)
    #     {
    #         'phi': 0.8,
    #         'fuel': {'CH4': 1.0},
    #         'oxidizer': {'O2': 0.21, 'N2': 0.79},
    #         'pressure': ct.one_atm,
    #         'temperature': 300.0,
    #
    #     },
    #     # Condition 2: Stoichiometric flame (phi=1.0)
    #     {
    #         'phi': 1.0,
    #         'fuel': {'CH4': 1.0},
    #         'oxidizer': {'O2': 0.21, 'N2': 0.79},
    #         'pressure': ct.one_atm,
    #         'temperature': 300.0
    #     },
    #     # Condition 3: Rich flame (phi=1.2)
    #     {
    #         'phi': 1.2,
    #         'fuel': {'CH4': 1.0},
    #         'oxidizer': {'O2': 0.21, 'N2': 0.79},
    #         'pressure': ct.one_atm,
    #         'temperature': 300.0
    #     }
    # ]
    
    # fitness = evaluate_reduction_mechanism('gri30.yaml', 'reduced_16sp.yaml', conditions_list)
    # print(f"Fitness of reduced mechanism: (fitness:.8f)")
    # higher value for better reduced mechanism
    
    condition = {
            'phi': 0.8,
            'fuel': {'CH4': 1.0},
            'oxidizer': {'O2': 0.21, 'N2': 0.79},
            'pressure': ct.one_atm,
            'temperature': 300.0,
            'mass_flow_rate': 0.04
        }
    results = run_premix_simulation('gri30.yaml', condition)
    print(results)