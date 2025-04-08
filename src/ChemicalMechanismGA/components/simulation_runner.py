import cantera as ct  
import numpy as np
import time

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
            
        # Validate the mechanism
        #self.validate_mechanism()
        
        self.reactor_type = reactor_type
        self.reactor = None
        self.reactor_network = None # used for time integration when useing IDT
        self.flame = None
        
    def validate_mechanism(self):
        try:
            # Set a simple state
            self.gas.TPX = 1000, 101325, {"CH4": 0.0775, "O2": 0.1938, "N2": 0.7287}

            # Create a constant-pressure reactor
            reactor = ct.IdealGasConstPressureReactor(self.gas)
            reactor_network = ct.ReactorNet([reactor])

            # Advance the simulation for a short time
            reactor_network.advance(1e-3)

            print("Mechanism validation passed.")
        except Exception as e:
            print(f"Mechanism validation failed: {e}")
            raise ValueError("Invalid mechanism provided.")
        
        
    def run_simulation(self, condition):
        # print(f"Condition: {condition}")
        fuel = condition['fuel']
        oxidizer_comp = condition['oxidizer']
        phi = condition['phi']
        pressure = condition['pressure']
        temperature = condition['temperature']
        mdot = condition['mass_flow_rate']
        
    
        mixture = {}
        # for fuel, fuel_val in fuel_comp.items():
        #     mixture[fuel] = fuel_val
        for ox, ox_val in oxidizer_comp.items():
            mixture[ox] = ox_val
        
        # Debugging: Print the constructed mixture
        # print(f"Mixture composition: {mixture}")
        # print(f"Species in gas: {self.gas.species_names}")
        self.gas.set_equivalence_ratio( phi, fuel, mixture)
        #float, float, dict
        self.gas.TP = temperature, pressure
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
        if self.reactor_type == "constant_pressure":
            print("Constant Pressure Reactor called")

            # Initialize the constant pressure reactor
            self.reactor = ct.IdealGasConstPressureReactor(self.gas)
            self.reactor_network = ct.ReactorNet([self.reactor])
            
            #self.reactor_network.preconditioner = ct.AdaptivePreconditioner()

            # Set tolerances
            self.reactor_network.rtol = 1e-5
            self.reactor_network.atol = 1e-8

            # Initialize time and time history
            time = 0.0
            end_time = 1.5 # in seconds
            time_step = 1e-4  # Fixed time step (e.g., 1e-3=1 ms)
            time_points = np.arange(0, end_time + time_step, time_step)  # Generate fixed time points
            time_history = {
                "time": [],
                "temperature": [],
                "pressure": [],
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
       
            
          
        elif self.reactor_type == "PREMIX":
            print("PREMIX reactor called")# equivalent to burner flame in cantera
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
                # Calculate burning distance for each species
                burning_distances = {}
                for i, species in enumerate(self.gas.species_names):
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
                
                results = {
                "grid": grid,
                "temperature_profile": T_profile,
                "mole_fractions": X_profiles,
                "heat_release_rate": flame.heat_release_rate,
                "max_temperature": np.max(T_profile),
                "flame_thickness": np.max(grid) - np.min(grid),
                "burning_distances": burning_distances  
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
                    "burning_distances" : {},
                    "error": str(e)
                 }
                
                
        elif self.reactor_type == "PSR":
            print("PSR reactor called")
            residence_time = 2  # s
            reactor_volume = 30.5 * (1e-2) ** 3  # m3

            # Simulation termination criterion
            max_simulation_time = 50  # seconds

            fuel_air_mixture_tank = ct.Reservoir(self.gas)
            exhaust = ct.Reservoir(self.gas)

            stirred_reactor = ct.IdealGasMoleReactor(self.gas, energy="on", volume=reactor_volume)

            mass_flow_controller = ct.MassFlowController(
                upstream=fuel_air_mixture_tank,
                downstream=stirred_reactor,
                mdot=stirred_reactor.mass / residence_time,
            )

            pressure_regulator = ct.PressureController(
                upstream=stirred_reactor,
                downstream=exhaust,
                primary=mass_flow_controller,
                K=1e-3,
            )
            reactor_network = ct.ReactorNet([stirred_reactor])
            # Create a SolutionArray to store the data
            time_history = ct.SolutionArray(self.gas, extra=["t"])
            # Set the maximum simulation time
            max_simulation_time = 50  # seconds
            # Start the stopwatch
            tic = time.time()
            # Set simulation start time to zero
            t = 0
            counter = 1
            try:
                while t < max_simulation_time:
                    t = reactor_network.step()

                    # Store every 10th value to reduce data size
                    if counter % 10 == 0:
                        # Append the state of the reactor to the time history
                        time_history.append(stirred_reactor.thermo.state, t=t)

                    counter += 1

                # Stop the stopwatch
                toc = time.time()
                print(f"Simulation took {toc - tic:3.2f}s to compute, with {counter} steps")

                # Extract results
                results = {
                    "time": time_history.t,  # Time points
                    "temperature_profile": time_history.T,  # Temperature profile
                    "mole_fractions": time_history.X.T,  # Species mole fractions (2D array: species × time points)
                    "max_temperature": np.max(time_history.T),  # Maximum temperature
                }

                # Add species-specific data for easier access
                for i, species in enumerate(self.gas.species_names):
                    results[species] = time_history.X[:, i]  # Mole fraction of each species over time
                #print("results PSR: ", results)
                return results

            except Exception as e:
                print(f"Error in PSR simulation: {e}")
                # Return empty profiles if simulation fails
                empty_array = np.array([0.0])
                return {
                    "time": empty_array,
                    "temperature_profile": empty_array,
                    "mole_fractions": np.zeros((len(self.gas.species_names), 1)),
                    "error": str(e),
                }
    
        
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