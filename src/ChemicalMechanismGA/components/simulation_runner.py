import cantera as ct  
import numpy as np
import time

class SimulationRunner:
    def __init__(self, mechanism, reactor_type):
        
        # Initialize the simulation runner with the mechanism and reactor type
        if isinstance(mechanism, ct.Solution):
            # If mechanism is already a Solution object, use it directly
            self.gas = mechanism
        else:
            # Otherwise, assume it's a file path and initialize a Solution object
            self.gas = ct.Solution(mechanism)
            
        self.reactor_type = reactor_type
        self.reactor = None
        self.reactor_network = None # used for time integration when useing IDT
        self.flame = None
        #self.gas.equilibrate('HP')
        #self.T_equi = self.gas.T
        
        
    def run_simulation(self, condition):
        # print(f"Condition: {condition}")
        fuel = condition['fuel']
        oxidizer_comp = condition['oxidizer']
        phi = condition['phi']
        pressure = condition['pressure']
        temperature = condition['temperature']
        # mdot = condition['mass_flow_rate']     
    
        # Debugging: Print the constructed mixture
        # print(f"Mixture composition: {mixture}")
        # print(f"Species in gas: {self.gas.species_names}")
        self.gas.set_equivalence_ratio( phi, fuel, oxidizer_comp)
        #float, float, dict
        self.gas.TP = temperature, pressure
        print(f"Initial Conditions: T={self.gas.T}, P={self.gas.P}, phi={phi}, mixture={oxidizer_comp}")
        
        # Initialize results dictionary with common data
        results = {
            "condition": condition.copy(),
            "species_names": self.gas.species_names,
            "initial_state": {
                "temperature": temperature,
                "mixture": oxidizer_comp.copy()
            }
        }
        
        # print(f"show results: {results}") #not printing...problem before this line
        if self.reactor_type == "constant_pressure":
            print("Constant Pressure Reactor called")

            # Initialize the constant pressure reactor
            self.reactor = ct.IdealGasConstPressureReactor(self.gas)
            self.reactor_network = ct.ReactorNet([self.reactor])
            
            # Set tolerances
            self.reactor_network.rtol = 1e-6
            self.reactor_network.atol = 1e-15

            # Initialize time and time history
            time = 0.0
            end_time = 0.05 # in seconds = 50ms
            time_step = 1e-5  # Fixed time step (e.g., 1e-5 = 0.01 ms)
            time_points = np.arange(0, end_time + time_step, time_step)  # Generate fixed time points
            time_history = {
                "time": [],
                "temperature": [],
                "mole_fractions": [],
            }
    
            # Run the simulation
            #while time < end_time:
            for t in time_points:
                try:
                    time = self.reactor_network.advance(t)
                except ct.CanteraError as e:
                    print(f"CanteraError: {e}")
                    break
                
                time_history["time"].append(time)
                time_history["temperature"].append(self.reactor.T)
                # time_history["pressure"].append(self.reactor.thermo.P)
                time_history["mole_fractions"].append(self.reactor.thermo.X.copy())

            # Store the time history for post-processing
            self.time_history = time_history
            
            # Calculate ignition delay time (IDT)
            time_array = np.array(time_history["time"])
            temperature_array = np.array(time_history["temperature"])
            # diff_temp = np.gradient(temperature_array,time_array)
            # ign_pos = np.argmax( diff_temp )
            # ignition_delay = time_array[ign_pos]
             
            # Prepare results
            results = {
                "time": time_array,
                "temperature_profile": temperature_array,
                # "pressure_profile": np.array(time_history["pressure"]),
                "mole_fractions": np.array(time_history["mole_fractions"]).T,
                #"ignition_delay": float(f"{ignition_delay:.6f}"),
                "max_temperature": np.max(temperature_array),
            }

            # Add species-specific mole fractions for easier access
            for i, species in enumerate(self.gas.species_names):
                results[species] = results["mole_fractions"][i, :]

            
            return results
            
          
       