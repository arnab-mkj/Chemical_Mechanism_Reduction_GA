import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# Define the gas mixture and initial conditions
gas = ct.Solution("gri30.yaml")  # Load GRI-Mech 3.0 mechanism
gas.TPX = 1000.0, ct.one_atm, "CH4:1.0, O2:1.0, N2:7.52"  # Initial temperature (K), pressure (atm), and composition

# Create a constant pressure reactor
reactor = ct.IdealGasConstPressureReactor(gas)

# Create a reactor network
sim = ct.ReactorNet([reactor])

# Time integration
time = 0.0  # Start time
end_time = 10  # End time (seconds)
time_step = 1e-5  # Time step (seconds)

# Arrays to store results
times = []
temperatures = []
heat_release_rates = []
species_concentrations = {species: [] for species in ["CH4", "O2", "CO2", "H2O", "N2"]}

# Run the simulation
while time < end_time:
    time = sim.step()  # Advance the simulation
    times.append(time)
    temperatures.append(reactor.T)
    # heat_release_rates.append(-reactor.heat_release_rate)  # Heat release rate (negative for exothermic reactions)
    for species in species_concentrations:
        species_concentrations[species].append(gas[species].X[0])  # Mole fraction of each species

# Plot 1: Temperature vs. Time
plt.figure(figsize=(8, 6))
plt.plot(times, temperatures, label="Temperature (K)", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs. Time in Constant Pressure Reactor")
plt.grid(True)
plt.legend()
plt.show()

# Plot 2: Species Concentrations vs. Time
plt.figure(figsize=(8, 6))
for species, concentrations in species_concentrations.items():
    plt.plot(times, concentrations, label=f"{species} Mole Fraction")
plt.xlabel("Time (s)")
plt.ylabel("Mole Fraction")
plt.title("Species Concentrations vs. Time in Constant Pressure Reactor")
plt.grid(True)
plt.legend()
plt.show()

# # Plot 3: Heat Release Rate vs. Time
# plt.figure(figsize=(8, 6))
# plt.plot(times, heat_release_rates, label="Heat Release Rate (W/m³)", color="blue")
# plt.xlabel("Time (s)")
# plt.ylabel("Heat Release Rate (W/m³)")
# plt.title("Heat Release Rate vs. Time in Constant Pressure Reactor")
# plt.grid(True)
# plt.legend()
# plt.show()