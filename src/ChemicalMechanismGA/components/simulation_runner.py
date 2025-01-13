import cantera as ct

class SimulationRunner:
    def __init__(self, mechanism_path, reactor_type="constant_pressure"):
        # Initialize the simulation Runner with the mechanisma nd Reactor type
        
        self.gas = ct.Solution(mechanism_path)
        self.reactor_type = reactor_type
        self.reactor = None
        self.reactor_network = None
        self.flame = None
        
    def set_initial_conditions(self, temperature, pressure, species_mole_fraction):
        
        #float, float, dict
        self.gas.TPX = temperature, pressure, species_mole_fraction
        
        if self.reactor_type == "batch":
            self.reactor = ct.IdealGasReactor(self.gas)
            self.reactor_network = ct.ReactorNet([self.reactor])
            
        elif self.reactor_type == "constant_pressure":
            self.reactor = ct.IdealGasConstPressureReactor(self.gas)
            self.reactor_network = ct.ReactorNet([self.reactor])
            
            
        elif self.reactor_type == "1D-flame":
            self.flame = ct.FreeFlame(self.gas)
            self.flame.set_initial_guess()  # ???????????
            
        else:
            raise ValueError(f"Unsupported reactor type: {self.reactor_type}")
        
        
    def run_simulation(self, end_time=1.0, time_step=1e-5):
        
        if self.reactor_type in ["batch", "constant_pressure"]:
            time = 0.0
            self.reactor_network.rtol = 1e-6  # Relative tolerance
            self.reactor_network.atol = 1e-15  # Absolute tolerance
            while time < end_time:
                time = self.reactor_network.step()
               # print(f"Time: {time:.5f}, Temperature: {self.reactor.T:.2f} K")
                
        elif self.reactor_type == "1D_flame":
            self.flame.solve(loglevel=1, refine_grid=True)
            print("1D flame simulation completed")
        
        else:
            raise ValueError(f"Unsupported reactor type : {self.reactor_type}")
        
    def get_results(self):
        
        if self.reactor_type in ["batch", "constant_pressure"]:
            return{
                "temperature": self.reactor.T,
                "pressure": self.reactor.thermo.P,
                "mole_fractions": self.reactor.thermo.X
            }
        elif self.reactor_type == "1D_flame":
            return{
                "temperature_profile": self.flame.T,
                "species_profiles": {species: self.flame.X[i,:] for i, species 
                                     in enumerate(self.gas.species_names)}
                
            }
        else:
            raise ValueError(f"Unsupported reactor type: {self.reactor_type}")