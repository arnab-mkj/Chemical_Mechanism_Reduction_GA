import cantera as ct  
import numpy as np
import time

class SimulationRunner:
    def __init__(self, mechanism, reactor_type):
        """
        Initialize a chemical reaction simulation runner.
        
        Args:
            mechanism (str or ct.Solution): Either a path to mechanism YAML file 
                                          or pre-initialized Cantera Solution object
            reactor_type (str): Type of reactor ('constant_pressure', etc.)
        """
        # Initialize the gas phase solution
        if isinstance(mechanism, ct.Solution):
            # Use existing Solution object if provided
            self.gas = mechanism
        else:
            # Create new Solution from mechanism file
            self.gas = ct.Solution(mechanism)
            
        self.reactor_type = reactor_type
        self.reactor = None          # Reactor instance
        self.reactor_network = None  # Reactor network for time integration
        self.flame = None            # Placeholder for flame simulations
        # Note: Equilibrium temperature calculation is commented out but available if needed
        # self.gas.equilibrate('HP')
        # self.T_equi = self.gas.T
        
    def run_simulation(self, condition):
        """
        Run a chemical reaction simulation with given initial conditions.
        
        Args:
            condition (dict): Dictionary containing:
                - fuel: Fuel specification
                - oxidizer: Oxidizer composition
                - phi: Equivalence ratio
                - pressure: Initial pressure [Pa]
                - temperature: Initial temperature [K]
                
        Returns:
            dict: Simulation results including:
                - Time profiles of temperature and species
                - Maximum temperature reached
                - Original condition data
        """
        # Extract simulation conditions
        fuel = condition['fuel']
        oxidizer_comp = condition['oxidizer']
        phi = condition['phi']
        pressure = condition['pressure']
        temperature = condition['temperature']
    
        # Set initial gas state
        self.gas.set_equivalence_ratio(phi, fuel, oxidizer_comp)
        self.gas.TP = temperature, pressure
        print(f"Initial Conditions: T={self.gas.T}, P={self.gas.P}, phi={phi}, mixture={oxidizer_comp}")
        
        # Initialize results dictionary with common data
        results = {
            "condition": condition.copy(),  # Store input conditions
            "species_names": self.gas.species_names,  # List of all species
            "initial_state": {
                "temperature": temperature,
                "mixture": oxidizer_comp.copy()
            }
        }
        
        if self.reactor_type == "constant_pressure":
            print("Constant Pressure Reactor called")

            # Initialize reactor and network
            self.reactor = ct.IdealGasConstPressureReactor(self.gas)
            self.reactor_network = ct.ReactorNet([self.reactor])
            
            # Set numerical tolerances
            self.reactor_network.rtol = 1e-6  # Relative tolerance
            self.reactor_network.atol = 1e-15  # Absolute tolerance

            # Simulation time parameters
            time = 0.0
            end_time = 0.05  # 50ms simulation time
            time_step = 1e-5  # 0.01ms fixed time step
            time_points = np.arange(0, end_time + time_step, time_step)
            
            # Initialize data storage
            time_history = {
                "time": [],             # Time points
                "temperature": [],       # Temperature history
                "mole_fractions": [],    # Species mole fractions
            }
    
            # Run the time integration
            for t in time_points:
                try:
                    time = self.reactor_network.advance(t)
                except ct.CanteraError as e:
                    print(f"CanteraError: {e}")
                    break
                
                # Store current state
                time_history["time"].append(time)
                time_history["temperature"].append(self.reactor.T)
                time_history["mole_fractions"].append(self.reactor.thermo.X.copy())

            # Store complete time history
            self.time_history = time_history
            
            # Convert to numpy arrays for analysis
            time_array = np.array(time_history["time"])
            temperature_array = np.array(time_history["temperature"])
            
            # Prepare final results dictionary
            results = {
                "time": time_array,
                "temperature_profile": temperature_array,
                "mole_fractions": np.array(time_history["mole_fractions"]).T,  # Transposed for species-wise access
                "max_temperature": np.max(temperature_array),
            }

            # Add individual species profiles for convenient access
            for i, species in enumerate(self.gas.species_names):
                results[species] = results["mole_fractions"][i, :]
            
            return results