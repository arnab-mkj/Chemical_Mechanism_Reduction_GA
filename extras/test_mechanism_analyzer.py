import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# Define the SimulationRunner class
class SimulationRunner:
    def __init__(self, mechanism):
        # Initialize the simulation runner with the mechanism
        self.gas = ct.Solution(mechanism)
        self.reactor = None
        self.reactor_network = None

    def run_constant_pressure_reactor(self, condition):
        # Extract simulation conditions
        fuel_comp = condition['fuel']
        oxidizer_comp = condition['oxidizer']
        pressure = condition['pressure']
        temperature = condition['temperature']
        end_time = condition.get("end_time")  # Default end time
        time_step = condition.get("time_step")
        phi = condition.get("phi")# Default time step

        # Create the mixture
        mixture = {}
        for fuel, fuel_val in fuel_comp.items():
            mixture[fuel] = fuel_val
        for ox, ox_val in oxidizer_comp.items():
            mixture[ox] = ox_val

        # Set the gas state
        self.gas.TP = temperature, pressure
        self.gas.set_equivalence_ratio( phi, fuel, oxidizer_comp )

        # Initialize the constant pressure reactor
        self.reactor = ct.IdealGasConstPressureReactor(self.gas)
        self.reactor_network = ct.ReactorNet([self.reactor])

        # Set tolerances
        self.reactor_network.rtol = 1e-6
        self.reactor_network.atol = 1e-15

        # Initialize time and time history
        time = 0.0
        time_history = {
            "time": [],
            "temperature": [],
            "pressure": [],
            "mole_fractions": [],
        }

        # Run the simulation
        while time < end_time:
            time = self.reactor_network.step()
            time_history["time"].append(time)
            time_history["temperature"].append(self.reactor.T)
            time_history["pressure"].append(self.reactor.thermo.P)
            time_history["mole_fractions"].append(self.reactor.thermo.X.copy())

        # Store the time history for post-processing
        self.time_history = time_history

        # Calculate ignition delay time (IDT)
        time_array = np.array(time_history["time"])
        temperature_array = np.array(time_history["temperature"])
        dT_dt = np.gradient(temperature_array, time_array)
        max_dT_dt_index = np.argmax(dT_dt)
        ignition_delay = time_array[max_dT_dt_index]

        # Prepare results
        results = {
            "time": time_array,
            "temperature_profile": temperature_array,
            "pressure_profile": np.array(time_history["pressure"]),
            "mole_fractions": np.array(time_history["mole_fractions"]).T,
            "ignition_delay": ignition_delay,
            "max_temperature": np.max(temperature_array),
        }

        # Add species-specific mole fractions for easier access
        for i, species in enumerate(self.gas.species_names):
            results[species] = results["mole_fractions"][i, :]

        print(f"Ignition delay time (IDT): {ignition_delay:.6e} seconds")
        return results

    def plot_results(self, results, species_to_plot=None):
        """
        Plot temperature profile and species concentrations.

        Args:
            results: Dictionary containing simulation results
            species_to_plot: List of species names to plot (if None, plots default species)
        """
        # Convert time to milliseconds for better readability
        time_ms = results['time'] * 1000

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot temperature profile
        ax1.plot(time_ms, results['temperature_profile'], 'b-', linewidth=2)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Profile')
        ax1.grid(True)

        # Mark the ignition delay time
        idt_ms = results['ignition_delay'] * 1000
        idt_temp = np.interp(idt_ms, time_ms, results['temperature_profile'])
        ax1.plot(idt_ms, idt_temp, 'ro', label=f'IDT: {idt_ms:.2f} ms')
        ax1.axvline(x=idt_ms, color='r', linestyle='--', alpha=0.3)
        ax1.legend()

        # Plot species concentrations
        if species_to_plot is None:
            species_to_plot = ['CH4', 'O2', 'CO2', 'H2O', 'OH']

        for species in species_to_plot:
            if species in results:
                ax2.plot(time_ms, results[species], label=species, linewidth=2)

        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Mole Fraction')
        ax2.set_title('Species Concentrations')
        ax2.grid(True)
        ax2.legend()

        # Use logarithmic scale for species concentrations if needed
        # ax2.set_yscale('log')  # Comment this line if you prefer linear scale

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

# Main function to run the simulation
if __name__ == "__main__":
    # Define the simulation conditions
    condition = {
            'phi': 0.8,
            'fuel': {'CH4': 1.0}, #mole fraction
            'oxidizer': {'O2': 0.21, 'N2': 0.79},
            'pressure': 101325, #* 0.0263, (20 torr)
            'temperature': 1800.0,
            'end_time': 0.001,  # End time in seconds
            'time_step': 1e-4,  # Time step in seconds
    }

    # Initialize the simulation runner
    runner = SimulationRunner(mechanism="gri30.yaml")

    # Run the constant pressure reactor simulation
    results = runner.run_constant_pressure_reactor(condition)

    # Print the results
    print("\n=== Simulation Results ===")
    print(f"Ignition Delay Time (IDT): {results['ignition_delay']:.6e} seconds")
    print(f"Maximum Temperature: {results['max_temperature']:.2f} K")
    # print("\nTemperature Profile:")
    # print(results["temperature_profile"])
    # print("\nPressure Profile:")
    # print(results["pressure_profile"])
    # print("\nMole Fractions of Key Species:")
    # for species in ["CH4", "O2", "CO2", "H2O", "OH"]:
    #     if species in results:
    #         print(f"{species}: {results[species]}")
            
    species_to_plot = ['CH4', 'O2', 'CO2', 'H2O', 'OH', 'CO']
    runner.plot_results(results, species_to_plot)