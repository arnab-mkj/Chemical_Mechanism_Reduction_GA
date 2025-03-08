import cantera as ct  
import numpy as np

class SimulationRunner:
    def __init__(self, mechanism, reactor_type):
        # Initialize the simulation Runner with the mechanisma nd Reactor type
        
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
        
    def run_simulation(self, condition):
        # print(f"Condition: {condition}")
        fuel_comp = condition['fuel']
        oxidizer_comp = condition['oxidizer']
        equivalence_ratio = condition['phi']
        pressure = condition['pressure']
        temperature = condition['temperature']
        mdot = condition['mass_flow_rate']
        
        # Debugging: Print the input conditions
        # print(f"Fuel composition: {fuel_comp}")
        # print(f"Oxidizer composition: {oxidizer_comp}")
        # print(f"Equivalence ratio: {equivalence_ratio}")
        # print(f"Pressure: {pressure}")
        # print(f"Temperature: {temperature}")
        mixture = {}
        for fuel, fuel_val in fuel_comp.items():
            mixture[fuel] = fuel_val
        for ox, ox_val in oxidizer_comp.items():
            mixture[ox] = ox_val
        
        # Debugging: Print the constructed mixture
        # print(f"Mixture composition: {mixture}")
        # print(f"Species in gas: {self.gas.species_names}")
        
        #float, float, dict
        self.gas.TPX = temperature, pressure, mixture
        # print(f"Gas state set: T={temperature}, P={pressure}, X={mixture}")
        
        # Initialize results dictionary with common data
        results = {
            "condition": condition.copy(),
            "species_names": self.gas.species_names,
            "initial_state": {
                "temperature": temperature,
                "pressure": pressure,
                "mixture": mixture.copy()
            }
        }
        # print(f"show results: {results}") #not printing...problem before this line
        if self.reactor_type == "batch":
            self.reactor = ct.IdealGasReactor(self.gas)
            self.reactor_network = ct.ReactorNet([self.reactor])
            # Work on this later, here IDT will come to play
            
        elif self.reactor_type == "constant_pressure":
            self.reactor = ct.ConstPressureReactor(contents=self.gas, energy='on')
            self.reactor_network = ct.ReactorNet([self.reactor])
            
          
        elif self.reactor_type == "PREMIX": # equivalent to burner flame in cantera
            #mdot = condition['mass_flow_rate']
            # Burner-stabilized flame
            self.gas.transport_model = 'mixture-averaged'  # or 'multi-component'
            # print(f"Transport model: {self.gas.transport_model}")
            flame = ct.BurnerFlame(self.gas, width=0.05)
            flame.burner.mdot = mdot  # g/(cm²·s)

            flame.transport_model = 'mixture-averaged'
            # print(f"transport model is: {flame.transport_model}")
            #Set simulation parameters
            flame.set_refine_criteria(ratio=3.0, slope=0.3, curve=1)
            # flame.max_time_step_count = 900
            
            flame.energy_enabled = True
            try:
                # Solve the flame
                flame.solve(loglevel=0, refine_grid=True)
                print("flame solved")  #debugging line
                flame.save('flame_fixed_T.csv', basis="mole", overwrite=True)
                flame.show_stats()
                
                # Extract grid and profiles
                grid = flame.grid
                T_profile = flame.T
                # Extract species profiles (2D array: species × grid points)
                X_profiles = flame.X
                # Collect results
                
                results = {
                "grid": grid,
                "temperature_profile": T_profile,
                "mole_fractions": X_profiles,
                "heat_release_rate": flame.heat_release_rate,
                "max_temperature": np.max(T_profile),
                "flame_thickness": np.max(grid) - np.min(grid)
                }
                # Add species-specific data for easier access
                for i, species in enumerate(self.gas.species_names):
                    results[species] = X_profiles[i, :] 
        # The key is the species name (e.g., 'H2', 'O2', etc.).
     #The value is a 1D array containing the mole fraction of that species at each grid point.
                #print("the results after run simulation", results)
                return results
            
            except Exception as e:
                print(f"Error in flame simulation: {e}")
                # Return empty profiles if simulation fails
                empty_array = np.array([0.0])
                return {
                    "grid": empty_array,
                    "temperature_profile": empty_array,
                    "mole_fractions": np.zeros((len(self.gas.species_names), 1)),
                    "error": str(e)
                 }
        else:
            raise ValueError(f"Unsupported reactor type: {self.reactor_type}")
        
    #region  
    # def run_simulation(self, end_time, time_step):
        
    #     if self.reactor_type in ["batch", "constant_pressure"]:
    #         time = 0.0
    #         self.reactor_network.rtol = 1e-6  # Relative tolerance
    #         self.reactor_network.atol = 1e-10  # Absolute tolerance
    #         #self.reactor_network.set_advance_limit(1e-3)  # Limit the maximum step size
    #         time_history = {
    #         "time": [],
    #         "temperature": [],
    #         "pressure": [],
    #         "species": []
    #         }
    #         while time < end_time:
    #             time += time_step
    #             #self.reactor_network.advance(time)
    #             time = self.reactor_network.step()
    #             # print(f"Time: {time:.5f}, Temperature: {self.reactor.T:.2f} K")
    #             # Store time history
    #             time_history["time"].append(time)
    #             time_history["temperature"].append(self.reactor.T)
    #             time_history["pressure"].append(self.reactor.thermo.P)
    #             time_history["species"].append(self.reactor.thermo.X.copy())
    #         self.time_history = time_history  # Store time history for post-processing
                
    #     elif self.reactor_type == "1D_flame":
    #         self.flame.solve(loglevel=1, refine_grid=True)
    #         print("1D flame simulation completed")
        
    #     else:
    #         raise ValueError(f"Unsupported reactor type : {self.reactor_type}")
    #endregion
        
    def calculate_ignition_delay(self):
        """
        Calculate ignition delay time based on the maximum temperature derivative.

        Returns:
            float: Ignition delay time (in seconds).
        """
        if not hasattr(self, "time_history"):
            raise ValueError("Time history data is not available. Run the simulation first.")

        time = np.array(self.time_history["time"])
        temperature = np.array(self.time_history["temperature"])

        # Calculate the temperature derivative
        dT_dt = np.gradient(temperature, time)

        # Find the time of maximum temperature derivative
        max_dT_dt_index = np.argmax(dT_dt)
        ignition_delay = time[max_dT_dt_index]

        print(f"Ignition delay time calculated: {ignition_delay} seconds")
        return ignition_delay
  
  
    def get_results(self):
        
        if self.reactor_type in ["batch", "constant_pressure"]:
            #ignition_delay = self.calculate_ignition_delay()
            return{
                "temperature": self.reactor.T,
                "pressure": self.reactor.thermo.P,
                "mole_fractions": self.reactor.thermo.X
                #"ignition_delay": ignition_delay
            }
        elif self.reactor_type == "1D_flame":
            return{
                "temperature_profile": self.flame.T,
                "species_profiles": {species: self.flame.X[i,:] for i, species 
                                     in enumerate(self.gas.species_names)}
                
            }
        else:
            raise ValueError(f"Unsupported reactor type: {self.reactor_type}")