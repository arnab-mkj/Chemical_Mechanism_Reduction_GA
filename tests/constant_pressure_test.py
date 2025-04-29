import cantera as ct
import numpy as np
import matplotlib.pyplot as plt


class MechanismComparison:
    def __init__(self, reduced_mech, full_mech):
        self.reduced_mech = reduced_mech
        self.full_mech = full_mech

    def run_simulation(self, mechanism, condition):
        """Run a constant-pressure reactor simulation for the given mechanism and condition."""
        try:
            gas = ct.Solution(mechanism)
            gas.TP = condition["temperature"], condition["pressure"]
            gas.set_equivalence_ratio(condition["phi"], condition["fuel"], condition["oxidizer"])

            # Reactor setup
            reactor = ct.IdealGasConstPressureReactor(gas)
            sim = ct.ReactorNet([reactor])

            # Time evolution
            time = []
            temperature = []
            species_mole_fractions = {species: [] for species in gas.species_names}

            t = 0.0
            while t < condition["end_time"]:
                t = sim.step()
                time.append(t)
                temperature.append(reactor.T)
                for species in gas.species_names:
                    species_mole_fractions[species].append(reactor.thermo[species].X)

            results = {
                "time": np.array(time),
                "temperature": np.array(temperature),
                "species_mole_fractions": {k: np.array(v) for k, v in species_mole_fractions.items()},
            }
            return results

        except Exception as e:
            print(f"Error in simulation for {mechanism}: {e}")
            return None

    def calculate_idt(self, results):
        """Calculate ignition delay time (IDT) based on the maximum temperature gradient."""
        try:
            time = results["time"]
            temperature = results["temperature"]
            dT_dt = np.gradient(temperature, time)
            idt_index = np.argmax(dT_dt)
            return time[idt_index]
        except Exception as e:
            print(f"Error in calculating IDT: {e}")
            return None

    def compare_mechanisms(self, conditions):
        """Compare the reduced mechanism with the full mechanism for the given conditions."""
        reduced_results_all = {}
        full_results_all = {}

        for phi in conditions["equivalence_ratios"]:
            condition = {
                "temperature": conditions["temperature"],
                "pressure": conditions["pressure"],
                "phi": phi,
                "fuel": conditions["fuel"],
                "oxidizer": conditions["oxidizer"],
                "end_time": conditions["end_time"],
            }

            print(f"Running simulations for equivalence ratio: {phi}")
            reduced_results = self.run_simulation(self.reduced_mech, condition)
            full_results = self.run_simulation(self.full_mech, condition)

            if reduced_results is None or full_results is None:
                print(f"Simulation failed for equivalence ratio: {phi}")
                continue

            reduced_results_all[phi] = reduced_results
            full_results_all[phi] = full_results

        # Plot temperature evolution for all equivalence ratios
        self.plot_temperature_evolution(reduced_results_all, full_results_all, conditions["equivalence_ratios"])

        # # Plot species mole fraction evolution for all equivalence ratios
        self.plot_species_evolution(reduced_results_all, full_results_all, conditions["equivalence_ratios"], conditions["key_species"])

        self.plot_IDT(reduced_results_all, full_results_all, conditions["equivalence_ratios"])
        # Calculate and print IDT for all equivalence ratios
        
    def plot_IDT(self, reduced_results_all, full_results_all, equivalence_ratios):
        """Plot IDT and temperature profiles for reduced and full mechanisms."""
        plt.figure(figsize=(10, 6))

        for phi in equivalence_ratios:
            if phi in reduced_results_all and phi in full_results_all:
                # Calculate IDT for reduced and full mechanisms
                reduced_idt = self.calculate_idt(reduced_results_all[phi])
                full_idt = self.calculate_idt(full_results_all[phi])

                # Extract temperature profiles and time
                reduced_time = reduced_results_all[phi]["time"]
                reduced_temp = reduced_results_all[phi]["temperature"]
                full_time = full_results_all[phi]["time"]
                full_temp = full_results_all[phi]["temperature"]

                # Plot temperature profiles
                plt.plot(reduced_time * 1000, reduced_temp, label=f"Reduced (phi={phi})", linestyle="--", color= 'green')
                plt.plot(full_time * 1000, full_temp, label=f"Full (phi={phi})", color = 'orange')

                # Mark IDT points
                plt.axvline(x=reduced_idt * 1000, color='blue', linestyle='--', linewidth='0.5',
                            label=f"Reduced IDT (phi={phi}): {reduced_idt*1000:.3e} ms")
                plt.axvline(x=full_idt * 1000, color='red', linestyle='--', linewidth='0.5',
                            label=f"Full IDT (phi={phi}): {full_idt*1000:.3e} ms")

                # Print IDT values
                print(f"IDT (Reduced Mechanism) for phi={phi}: {reduced_idt*1000:.3e} ms")
                print(f"IDT (Full Mechanism) for phi={phi}: {full_idt*1000:.3e} ms")

        # Add labels, title, and legend
        plt.xlabel('Time [ms]', fontsize=12)
        plt.ylabel('Temperature [K]', fontsize=12)
        plt.title('Temperature Profiles with Ignition Delay Times (IDT)', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlim(0.0, 0.05)
        plt.savefig(f'E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/Run_8/IDT')
        # Adjust layout for better appearance
        plt.tight_layout()
        plt.show()
        
        
        
    def plot_temperature_evolution(self, reduced_results_all, full_results_all, equivalence_ratios):
        """Plot temperature evolution with time for all equivalence ratios."""
        plt.figure(figsize=(10, 6))
        for phi in equivalence_ratios:
            if phi in reduced_results_all and phi in full_results_all:
                plt.plot(reduced_results_all[phi]["time"]*1000, reduced_results_all[phi]["temperature"],
                         label=f"Reduced (phi={phi})",color='green', linestyle="--")
                plt.plot(full_results_all[phi]["time"]*1000, full_results_all[phi]["temperature"],
                         label=f"Full (phi={phi})", color='orange')
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Evolution for Different Equivalence Ratios")
        plt.xlim(0.0, 0.05)
        plt.legend()
        plt.savefig(f'E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/Run_8/temperature')
        plt.grid()
        plt.show()

    def plot_species_evolution(self, reduced_results_all, full_results_all, equivalence_ratios, key_species):
        """Plot species mole fraction evolution with time for all equivalence ratios."""
        for species in key_species:
            plt.figure(figsize=(10, 6))
            for phi in equivalence_ratios:
                if phi in reduced_results_all and phi in full_results_all:
                    plt.plot(reduced_results_all[phi]["time"]*1000, reduced_results_all[phi]["species_mole_fractions"][species],
                             label=f"{species} Reduced (phi={phi})", color='green', linestyle="--")
                    plt.plot(full_results_all[phi]["time"]*1000, full_results_all[phi]["species_mole_fractions"][species],
                             label=f"{species} Full (phi={phi})", color='orange')
            plt.xlabel("Time (ms)")
            plt.ylabel("Mole Fraction")
            plt.title(f"{species} Mole Fraction Evolution for Different Equivalence Ratios")
            plt.xlim(0.0, 0.05)
            plt.legend()
            plt.savefig(f'E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/Run_8/mole_fraction_{species}')
            plt.grid()
            plt.show()


# Example Usage
if __name__ == "__main__":
    # Define conditions
    conditions = {
        "temperature": 2561,  # Initial temperature in K
        "pressure":  ct.one_atm,  # Pressure in Pa
        "equivalence_ratios": [0.4],  # Equivalence ratios to test
        "fuel": "CH4",  # Fuel
        "oxidizer": {"O2": 1.0, "N2": 3.76},  # Oxidizer composition
        "end_time": 0.01,  # End time for the simulation in seconds
        "key_species": ["CH4", "O2", "CO2", "CO", "H2O"],  # Species to plot
    }

    # Mechanism files
    reduced_mech = "E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/Run_8/reduced_mech_65_rxns.yaml"
    full_mech = "gri30.yaml"

    # Initialize and run comparison
    comparison = MechanismComparison(reduced_mech, full_mech)
    comparison.compare_mechanisms(conditions)