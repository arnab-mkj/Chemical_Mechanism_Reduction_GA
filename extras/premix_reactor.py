import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

class PremixFlameComparison:
    def __init__(self, gas, reduced_mech, gri_mech):
        self.gas = gas
        self.reduced_mech = reduced_mech
        self.gri_mech = gri_mech

    def run_premix_flame(self, mechanism, condition):
        """Run a premixed flame simulation for the given mechanism."""
        try:
            gas = ct.Solution(mechanism, transport_model="mixture-averaged")
            gas.TP = condition["temperature"], condition["pressure"]
            gas.set_equivalence_ratio(condition["phi"], condition["fuel"], condition["oxidizer"])
            
            # Burner-stabilized flame
            flame = ct.BurnerFlame(gas, width=0.01)
            flame.burner.mdot = condition["mass_flow_rate"]
            flame.set_refine_criteria(ratio=3.0, slope=0.07, curve=0.14)
            flame.energy_enabled = True

            # Solve the flame
            flame.solve(loglevel=0, auto=True)
            profile = flame.to_array()
            X_profiles = flame.X
            grid = flame.grid
            T_profile = flame.T
            burning_distances = {}
        
            for i, species in enumerate(gas.species_names):
                mole_fraction_profile = X_profiles[i, :]
                # Find the index where the mole fraction drops below a threshold (e.g., 1% of max)
                max_mole_fraction = np.max(mole_fraction_profile)
                threshold = 0.01 * max_mole_fraction  # 1% of the maximum mole fraction
                burning_indices = np.where(mole_fraction_profile > threshold)[0]

                if len(burning_indices) > 0:
                    # Burning distance is the distance between the first and last significant points
                    burning_distance = grid[burning_indices[-1]] - grid[burning_indices[0]]
                else:
                    # If the species is not consumed significantly, set burning distance to 0
                    burning_distance = 0.0

                burning_distances[species] = burning_distance

            # Extract results
            results = {
                "grid": grid,
                "temperature_profile": T_profile,
                "burning_velocity": flame.velocity[0],
                "heat_release_rate": flame.heat_release_rate,
                "burning_distances": burning_distances
            }
                    
            for i, species in enumerate(gas.species_names):
                results[species] = X_profiles[i, :] 
                    
            return results, profile

        except Exception as e:
            print(f"Error in premixed flame simulation for {mechanism}: {e}")
            return None, None

    def run_constant_pressure(self, mechanism, condition):
        """Run a constant pressure reactor simulation for the given mechanism."""
        try:
            gas = ct.Solution(mechanism)
            gas.TP = condition["temperature"], condition["pressure"]
            gas.set_equivalence_ratio(condition["phi"], condition["fuel"], condition["oxidizer"])
            
            # Setup constant pressure reactor
            reactor = ct.IdealGasConstPressureReactor(gas)
            sim = ct.ReactorNet([reactor])
            
            # Simulation parameters
            times = np.linspace(0, 0.1, 1000)  # Simulate for 0.1 seconds
            temp_profiles = []
            species_profiles = {}
            heat_release_rates = []
            
            for species in ["CH4", "O2", "CO2", "CO", "OH"]:
                species_profiles[species] = []
            
            # Run simulation
            for t in times:
                sim.advance(t)
                temp_profiles.append(reactor.T)
                heat_release_rates.append(reactor.thermo.net_production_rates.sum() * -reactor.thermo.enthalpy_mass)
                for species in species_profiles:
                    species_profiles[species].append(reactor.thermo[species].X[0])
            
            # Estimate burning velocity (approximation for constant pressure reactor)
            # Using ignition delay time to approximate reactivity
            ignition_delay = 0
            for i in range(1, len(temp_profiles)):
                if temp_profiles[i] > temp_profiles[0] + 400:  # Temperature rise of 400 K
                    ignition_delay = times[i]
                    break
            burning_velocity = 1.0 / ignition_delay if ignition_delay > 0 else 0.0  # Simplified approximation
            
            # Create a profile object for consistency
            class Profile:
                def __init__(self, grid, T, X):
                    self.grid = grid
                    self.T = T
                    self.X = X
            
            X_profiles = np.array([species_profiles[sp] for sp in species_profiles.keys()])
            profile = Profile(times, np.array(temp_profiles), X_profiles)
            
            results = {
                "grid": times,
                "temperature_profile": np.array(temp_profiles),
                "burning_velocity": burning_velocity,
                "heat_release_rate": np.array(heat_release_rates)
            }
            
            for species, profile in species_profiles.items():
                results[species] = np.array(profile)
            
            return results, profile

        except Exception as e:
            print(f"Error in constant pressure simulation for {mechanism}: {e}")
            return None, None

    def compare_mechanisms(self, condition, reactor_type="premixed_flame"):
        """Compare the reduced mechanism with GRI-Mech 3.0."""
        # Select the simulation type based on reactor_type
        if reactor_type == "premixed_flame":
            run_simulation = self.run_premix_flame
        else:  # constant_pressure
            run_simulation = self.run_constant_pressure

        # Run simulations for both mechanisms
        reduced_results, reduced_profile = run_simulation(self.reduced_mech, condition)
        gri_results, gri_profile = run_simulation(self.gri_mech, condition)

        if reduced_results is None or gri_results is None:
            return None

        # Plot results and return summary
        return self.plot_results(reduced_results, gri_results, reduced_profile, gri_profile, reactor_type)

    def plot_results(self, reduced_results, gri_results, reduced_profile, gri_profile, reactor_type):
        """Generate comparison plots and return summary statistics."""
        # Adjust x-axis label based on reactor type
        x_label = "Distance (cm)" if reactor_type == "premixed_flame" else "Time (s)"
        x_scale = 100 if reactor_type == "premixed_flame" else 1  # Convert to cm for premixed flame
        
        # Plot 1: Temperature Profile
        plt.figure(figsize=(10, 6))
        plt.plot(reduced_profile.grid * x_scale, reduced_profile.T, label="Reduced Mechanism", color="blue")
        plt.plot(gri_profile.grid * x_scale, gri_profile.T, label="GRI-Mech 3.0", color="red")
        plt.xlabel(x_label)
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Profile")
        plt.xlim(0.0, 0.2 if reactor_type == "premixed_flame" else 0.1)
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig("temperature_profile.png")
        plt.close()

        # Plot 2: Mole Fraction Profiles
        species_to_plot = ["CH4", "O2", "CO2", "CO", "OH"]
        species_colors = {
            "CH4": "tab:blue",
            "O2": "tab:orange",
            "CO2": "tab:green",
            "CO": "tab:red",
            "OH": "tab:purple"
        }
        
        plt.figure(figsize=(10, 6))
        for species in species_to_plot:
            if species in reduced_results and species in gri_results:
                color = species_colors[species]
                reduced_X = reduced_results[species]
                gri_X = gri_results[species]
                
                plt.plot(reduced_results["grid"] * x_scale, reduced_X, 
                         label=f"{species} (Reduced)", linestyle="--", color=color)
                plt.plot(gri_results["grid"] * x_scale, gri_X, 
                         label=f"{species} (GRI-Mech 3.0)", linestyle="-", color=color)
        
        plt.xlabel(x_label)
        plt.ylabel("Mole Fraction")
        plt.title("Mole Fraction Profiles")
        plt.xlim(0.0, 0.8 if reactor_type == "premixed_flame" else 0.1)
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig("mole_fraction_profiles.png")
        plt.close()

        # Plot 3: Heat Release Rate
        plt.figure(figsize=(10, 6))
        plt.plot(reduced_results["grid"] * x_scale, reduced_results["heat_release_rate"], 
                 label="Reduced Mechanism", color="blue")
        plt.plot(gri_results["grid"] * x_scale, gri_results["heat_release_rate"], 
                 label="GRI-Mech 3.0", color="red")
        plt.xlabel(x_label)
        plt.ylabel("Heat Release Rate (W/m³)")
        plt.title("Heat Release Rate Profile")
        plt.xlim(0.0, 0.8 if reactor_type == "premixed_flame" else 0.1)
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig("heat_release_rate.png")
        plt.close()

        # Return summary statistics
        return {
            "reduced_burning_velocity": reduced_results["burning_velocity"],
            "gri_burning_velocity": gri_results["burning_velocity"],
            "reduced_max_temp": max(reduced_results["temperature_profile"]),
            "gri_max_temp": max(gri_results["temperature_profile"])
        }