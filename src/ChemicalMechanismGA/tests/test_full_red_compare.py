import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

class MechanismComparison:
    def __init__(self, file_path, reduced_mech, full_mech, xlim):
        self.reduced_mech = reduced_mech
        self.full_mech = full_mech
        self.file_path = file_path
        self.xlim = xlim

    def run_simulation(self, mechanism, condition):
        """Run a constant-pressure reactor simulation for the given mechanism and condition."""
        try:
            gas = ct.Solution(mechanism)
            gas.TP = condition["temperature"], condition["pressure"]
            gas.set_equivalence_ratio(condition["phi"], condition["fuel"], condition["oxidizer"], basis='mass')

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
            return None

    def calculate_idt(self, results):
        """Calculate ignition delay time (IDT) based on the maximum temperature gradient."""
        try:
            time = results["time"]
            temperature = results["temperature"]
            dT_dt = np.gradient(temperature, time)
            idt_index = np.argmax(dT_dt)
            return time[idt_index]
        except Exception:
            return None

    def compare_mechanisms(self, conditions):
        """Compare the reduced mechanism with the full mechanism for the given conditions."""
        reduced_results_all = {}
        full_results_all = {}
        idts = {}
        max_temps = {}

        for phi in conditions["equivalence_ratios"]:
            condition = {
                "temperature": conditions["temperature"],
                "pressure": conditions["pressure"],
                "phi": phi,
                "fuel": conditions["fuel"],
                "oxidizer": conditions["oxidizer"],
                "end_time": conditions["end_time"],
            }

            reduced_results = self.run_simulation(self.reduced_mech, condition)
            full_results = self.run_simulation(self.full_mech, condition)

            if reduced_results is None or full_results is None:
                continue

            reduced_results_all[phi] = reduced_results
            full_results_all[phi] = full_results

            # Calculate IDT and max temperature
            reduced_idt = self.calculate_idt(reduced_results)
            full_idt = self.calculate_idt(full_results)
            reduced_max_temp = np.max(reduced_results["temperature"]) if reduced_results else 0
            full_max_temp = np.max(full_results["temperature"]) if full_results else 0

            idts[phi] = {
                "reduced": reduced_idt * 1000 if reduced_idt is not None else None,  # Convert to ms
                "full": full_idt * 1000 if full_idt is not None else None
            }
            max_temps[phi] = {
                "reduced": reduced_max_temp,
                "full": full_max_temp
            }

        if not reduced_results_all or not full_results_all:
            return None

        # Plot results
        self.plot_temperature_evolution(reduced_results_all, full_results_all, conditions["equivalence_ratios"])
        self.plot_species_evolution(reduced_results_all, full_results_all, conditions["equivalence_ratios"], conditions["key_species"])
        self.plot_IDT(reduced_results_all, full_results_all, conditions["equivalence_ratios"])

        return {
            "idts": idts,
            "max_temps": max_temps
        }

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
                plt.plot(reduced_time * 1000, reduced_temp, label=f"Reduced (phi={phi})", linestyle="--", color='blue')
                plt.plot(full_time * 1000, full_temp, label=f"Full (phi={phi})", color='red')

                # Mark IDT points if they exist
                if reduced_idt is not None:
                    plt.axvline(x=reduced_idt * 1000, color='blue', linestyle='--', linewidth='0.5',
                                label=f"Reduced IDT (phi={phi}): {reduced_idt*1000:.6f} ms")
                if full_idt is not None:
                    plt.axvline(x=full_idt * 1000, color='red', linestyle='--', linewidth='0.5',
                                label=f"Full IDT (phi={phi}): {full_idt*1000:.6f} ms")

        # Add labels, title, and legend
        plt.xlabel('Time [ms]', fontsize=12)
        plt.ylabel('Temperature [K]', fontsize=12)
        plt.title('Temperature Profiles with Ignition Delay Times (IDT)', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlim(0.0, self.xlim)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{self.file_path}/IDT.png')
        plt.close()

    def plot_temperature_evolution(self, reduced_results_all, full_results_all, equivalence_ratios):
        """Plot temperature evolution with time for all equivalence ratios."""
        plt.figure(figsize=(10, 6))
        for phi in equivalence_ratios:
            if phi in reduced_results_all and phi in full_results_all:
                plt.plot(reduced_results_all[phi]["time"]*1000, reduced_results_all[phi]["temperature"],
                         label=f"Reduced (phi={phi})", color='blue', linestyle="--")
                plt.plot(full_results_all[phi]["time"]*1000, full_results_all[phi]["temperature"],
                         label=f"Full (phi={phi})", color='red')
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Evolution for Different Equivalence Ratios")
        plt.xlim(0.0, self.xlim)
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(f'{self.file_path}/temperature.png')
        plt.close()

    def plot_species_evolution(self, reduced_results_all, full_results_all, equivalence_ratios, key_species):
        """Plot species mole fraction evolution with time for all equivalence ratios."""
        for species in key_species:
            plt.figure(figsize=(10, 6))
            for phi in equivalence_ratios:
                if phi in reduced_results_all and phi in full_results_all:
                    if species in reduced_results_all[phi]["species_mole_fractions"] and species in full_results_all[phi]["species_mole_fractions"]:
                        plt.plot(reduced_results_all[phi]["time"]*1000, reduced_results_all[phi]["species_mole_fractions"][species],
                                 label=f"{species} Reduced (phi={phi})", color='blue', linestyle="--")
                        plt.plot(full_results_all[phi]["time"]*1000, full_results_all[phi]["species_mole_fractions"][species],
                                 label=f"{species} Full (phi={phi})", color='red')
            plt.xlabel("Time (ms)")
            plt.ylabel("Mole Fraction")
            plt.title(f"{species} Mole Fraction Evolution for Different Equivalence Ratios")
            plt.xlim(0.0, self.xlim)
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig(f'{self.file_path}/mole_fraction_{species}.png')
            plt.close()