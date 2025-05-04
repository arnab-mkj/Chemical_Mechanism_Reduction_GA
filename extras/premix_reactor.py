
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
            print(f"Error in flame simulation for {mechanism}: {e}")
            return None



    def compare_mechanisms(self, condition):
        """Compare the reduced mechanism with GRI-Mech 3.0."""
        # Run simulations for both mechanisms
        reduced_results, reduced_profile = self.run_premix_flame(self.reduced_mech, condition)
        gri_results, gri_profile = self.run_premix_flame(self.gri_mech, condition)

        if reduced_results is None or gri_results is None:
            print("Simulation failed for one or both mechanisms.")
            return

        # Plot results
        self.plot_results(reduced_results, gri_results, reduced_profile, gri_profile)



    def plot_results(self, reduced_results, gri_results,reduced_profile, gri_profile):
        """Generate comparison plots."""
        grid_reduced = reduced_results["grid"]
        grid_gri = gri_results["grid"]

        # Plot 1: Temperature Profile
        plt.figure(figsize=(10, 6))
        plt.plot(reduced_profile.grid * 100, reduced_profile.T, label="Reduced Mechanism", color="blue")
        plt.plot(gri_profile.grid * 100, gri_profile.T, label="GRI-Mech 3.0", color="red")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Profile")
        plt.xlim(0.0, 0.2) # 5cm
        plt.legend()
        plt.grid()
        plt.show()

        # Plot 2: Burning Velocity
        print(f"Burning Velocity (Reduced): {reduced_results['burning_velocity']} m/s")
        print(f"Burning Velocity (GRI-Mech 3.0): {gri_results['burning_velocity']} m/s")

        # Plot 3: Mole Fraction Profiles
        species_to_plot = ["CH4", "O2", "CO2", "CO", "OH"]

# Assign a fixed color to each species
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
                plt.plot(reduced_profile.grid * 100, reduced_profile(f"{species}").X, label=f"{species} (Reduced)", linestyle="--", color=color)
                plt.plot(gri_profile.grid * 100, reduced_profile(f"{species}").X, label=f"{species} (GRI-Mech 3.0)", linestyle="-", color=color)
        plt.xlabel("Distance (cm)")
        plt.ylabel("Mole Fraction")
        plt.title("Mole Fraction Profiles")
        plt.xlim(0.0, 0.2) # in cm
        plt.legend()
        plt.grid()
        plt.show()

        # Plot 4: Heat Release Rate
        plt.figure(figsize=(10, 6))
        plt.plot(grid_reduced*100, reduced_results["heat_release_rate"], label="Reduced Mechanism", color="blue")
        plt.plot(grid_gri*100, gri_results["heat_release_rate"], label="GRI-Mech 3.0", color="red")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Heat Release Rate (W/m³)")
        plt.title("Heat Release Rate Profile")
        plt.xlim(0.0, 0.2) # in cm
        plt.legend()
        plt.grid()
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Define conditions
    condition = {
        "phi": 1.0,
        "temperature": 1000,  # Initial temperature in K
        "pressure": ct.one_atm,  # Pressure in Pa
        "fuel": "CH4",  # Fuel
        "oxidizer": {"O2": 0.21, "N2": 0.79},
        'mass_flow_rate': 0.04# Oxidizer composition
    }
    file_path = "E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/absolute"
    # Mechanism files
    reduced_mech = "reduced_mech_64_rxns.yaml"
    gri_mech = "gri30.yaml"

    # Initialize and run comparison
    gas = ct.Solution(gri_mech)
    comparison = PremixFlameComparison(gas, f"{file_path}/{reduced_mech}", gri_mech)
    comparison.compare_mechanisms(condition)