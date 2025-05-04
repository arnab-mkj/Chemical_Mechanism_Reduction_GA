import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import random
import json

def create_reduced_mechanism(full_mech_file, reduction_fraction, reduced_mech_file):
    gas = ct.Solution(full_mech_file)
    n_reactions = gas.n_reactions

    duplicate_groups = {}
    non_duplicates = []
    for i in range(n_reactions):
        r = gas.reaction(i)
        if r.duplicate:
            eq = r.equation
            if eq not in duplicate_groups:
                duplicate_groups[eq] = []
            duplicate_groups[eq].append(i)
        else:
            non_duplicates.append(i)

    total_reactions_to_remove = int(reduction_fraction * n_reactions)
    remove_from_non_dup = sorted(random.sample(non_duplicates, min(total_reactions_to_remove, len(non_duplicates))))
    keep_indices = [i for i in range(n_reactions) if i not in remove_from_non_dup]

    remaining_to_remove = total_reactions_to_remove - len(remove_from_non_dup)
    if remaining_to_remove > 0:
        duplicate_keys = list(duplicate_groups.keys())
        random.shuffle(duplicate_keys)
        removed_dup_count = 0
        for key in duplicate_keys:
            group = duplicate_groups[key]
            if removed_dup_count + len(group) <= remaining_to_remove:
                for idx in group:
                    if idx in keep_indices:
                        keep_indices.remove(idx)
                removed_dup_count += len(group)
            if removed_dup_count >= remaining_to_remove:
                break

    species = gas.species()
    reactions = [gas.reaction(i) for i in keep_indices]
    reduced_gas = ct.Solution(thermo='ideal-gas', kinetics='GasKinetics', transport='mixture-averaged',
                              species=species, reactions=reactions)
    reduced_gas.write_yaml(reduced_mech_file)
    print(f"[INFO] Reduced mechanism saved to {reduced_mech_file}")
    return reduced_gas.n_reactions, gas.n_reactions

def run_simulation(mechanism_file, T0, P0, phi, species_list, t_end=0.05):
    gas = ct.Solution(mechanism_file)
    gas.TP = T0, P0
    gas.set_equivalence_ratio(phi, fuel="CH4", oxidizer="O2:0.21, N2:0.79")
    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])

    time_step = 1e-6
    time_points = np.arange(0, t_end + time_step, time_step)
    temps, times = [], []
    mole_fractions = {sp: [] for sp in species_list}

    for t in time_points:
        sim.advance(t)
        times.append(sim.time)
        temps.append(reactor.T)
        for sp in species_list:
            idx = gas.species_index(sp)
            mole_fractions[sp].append(reactor.thermo.X[idx])

    return np.array(times), np.array(temps), mole_fractions



def calculate_idt(time, temp):
    try:
        dTdt = np.gradient(temp, time)
        return time[np.argmax(dTdt)]
    except Exception as e:
        print(f"IDT calculation failed: {e}")
        return None
    
def sigmoid_norm(x, scale):
    return 1 / (1 + np.exp(-scale * x))

def log_norm(x, scale):
    return np.log(1 + scale * x) 

def print_fitness_breakdown(name, full_vals, red_vals, time_vals=None, max_samples=10, scale=10):
    """
    Enhanced fitness breakdown function that:
    - For time-series data: shows samples + integrated errors
    - For single values (like IDT): shows direct comparison
    """
    full_vals = np.array(full_vals)
    red_vals = np.array(red_vals)
    
    # Check if this is single value (like IDT) or time-series
    is_time_series = len(full_vals) > 1 and time_vals is not None
    
    print(f"\n=== {name} Fitness Breakdown ===")
    
    if is_time_series:
        # Time-series data (temperature, species profiles)
        sq_diff = abs(red_vals - full_vals) ** 2
        lin_diff = np.sqrt((red_vals - full_vals) ** 2)
        
        step = max(1, len(time_vals) // 1000)
        sample_indices = range(0, len(time_vals), step)
        # Show sample points
        print("\n 10 Sample Points (spaced every {step} points):")
        
        
        for i in sample_indices[:10]:
            print(f"t={time_vals[i]:.4e}: Full={full_vals[i]:.4e} | Reduced={red_vals[i]:.4e} | "
                  f"Diff={lin_diff[i]:.4e} | SqDiff={sq_diff[i]:.4e}")
        
        # Calculate integrated metrics
        try:
            integrated_lin = simpson(lin_diff)
            integrated_sq = simpson(sq_diff)
            lin_error = integrated_sq/simpson(np.sqrt(full_vals**2))
            sq_error = integrated_lin/simpson((full_vals**2))
            
        except:
            integrated_lin = integrated_sq = lin_error = np.inf
        
        print("\nIntegrated Error Metrics:")
        print(f"Integrated Linear Difference: {integrated_lin:.6e}")
        print(f"Integrated Squared Difference: {integrated_sq:.6e}")
        print(f" Integrated Linear Error: {lin_error:.6e}")
        print(f" Integrated Squared Error: {sq_error:.6e}")
        
    else:
        # Single value comparison (like IDT)
        lin_diff = np.sqrt((red_vals[0] - full_vals[0])**2)
        sq_diff = (red_vals[0] - full_vals[0])**2
        
        lin_error = lin_diff/np.sqrt(full_vals[0]**2) if full_vals[0] != 0 else np.inf
        sq_error  = sq_diff/((full_vals[0])**2)
        
        print("\nDirect Comparison:")
        print(f"Full Value: {full_vals[0]:.6e}")
        print(f"Reduced Value: {red_vals[0]:.6e}")
        print(f"Linear Difference: {lin_diff:.6e}")
        print(f"Squared Difference: {sq_diff:.6e}")
        print(f"Linear Error: {lin_error:.6e}")
        print(f"Squared Error: {sq_error:.6e}")
    
    # Calculate normalized scores (using relative error)
    sig_norm_lin = sigmoid_norm(lin_error, scale=scale)
    log_norm_lin = log_norm(lin_error, scale=scale)
    sig_norm_sq = sigmoid_norm(sq_error, scale=scale)
    log_norm_sq= log_norm(sq_error, scale=scale)
    
    print("\nNormalized Scores:")
    print(f"Sigmoid Normalized Linear: {sig_norm_lin:.6f}")
    print(f"Log Normalized Linear: {log_norm_lin:.6f}")
    print(f"Sigmoid Normalized Squared: {sig_norm_sq:.6f}")
    print(f"Log Normalized Squared: {log_norm_sq:.6f}")

def run_test(params_file="params_test.json"):
    # Load parameters from params_test.json
    with open(params_file, "r") as f:
        config = json.load(f)
    
    full_mech = config["full_mech"]
    reduced_mech = config["reduced_mech"]
    reduction_fraction = config["reduction_fraction"]
    temp = config["temperature"]
    pres = config["pressure"]
    phi = config["equivalence_ratio"]
    fuel = config["fuel"]
    oxidizer_comp = config["oxidizer"]
    key_species = config["key_species"]

    # Create reduced mechanism
    reduced_count, full_count = create_reduced_mechanism(full_mech, reduction_fraction, reduced_mech)

    # Run simulations
    time_full, temp_full, mole_full = run_simulation(full_mech, temp, pres, phi, key_species)
    time_red, temp_red, mole_red = run_simulation(reduced_mech, temp, pres, phi, key_species)

    # Calculate IDT
    idt_full = calculate_idt(time_full, temp_full)
    idt_red = calculate_idt(time_red, temp_red)

    # Print fitness breakdown
    print("\n=== Quantitative Fitness Breakdown ===")
    print_fitness_breakdown("Temperature Profile", temp_full, temp_red, time_full)
    print_fitness_breakdown("Ignition Delay Time", [idt_full], [idt_red])
    for sp in key_species:
        print_fitness_breakdown(f"{sp} Mole Fraction", mole_full[sp], mole_red[sp], time_full)

    print(f"\nReactions kept: {reduced_count}/{full_count} ({100 * reduced_count / full_count:.2f}%)")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(time_full * 1000, temp_full, label="Full", linewidth=2)
    plt.plot(time_red * 1000, temp_red, label="Reduced", linestyle="--")
    plt.xlabel("Time [ms]")
    plt.ylabel("Temperature [K]")
    plt.title("Temperature Profile")
    plt.xlim(0.0, 0.4)
    plt.legend()
    plt.grid(True)
    plt.savefig("fitness_temp_profile.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    for sp in key_species:
        plt.plot(time_full * 1000, mole_full[sp], label=f"{sp} Full")
        plt.plot(time_red * 1000, mole_red[sp], linestyle="--", label=f"{sp} Reduced")
    plt.xlabel("Time [ms]")
    plt.ylabel("Mole Fraction")
    plt.title("Species Mole Fraction Profiles")
    plt.xlim(0.0, 0.4)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.grid(True)
    plt.savefig("fitness_mole_fraction.png")
    plt.close()

    return "Fitness test completed. Plots saved to 'fitness_temp_profile.png' and 'fitness_mole_fraction.png'"


if __name__ == "__main__":
    run_test()