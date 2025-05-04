import sys
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import timeit
import json
import os
from pathlib import Path

class MechanismReducer:
    def __init__(self, config_path="params.json"):
        """
        Initializes a chemical mechanism reducer that creates simplified versions
        of detailed reaction mechanisms by removing less important reactions.
        
        The reducer uses sensitivity analysis to identify and remove reactions
        that have minimal impact on ignition delay time predictions.
        
        Args:
            config_path (str): Path to JSON configuration file containing:
                - mechanism: Path to detailed mechanism file
                - conditions: Simulation conditions (T, P, equivalence ratio)
                - key_species: Important species to track
                - output_directory: Where to save reduced mechanisms
        """
        # Load configuration from JSON file
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Get list of key species to monitor from config
        self.key_species = self.config.get("key_species", [])
        
        # Handle both single condition (dict) and multiple conditions (list)
        conditions = self.config.get("conditions", {})
        if isinstance(conditions, list):
            # Use first condition if multiple are provided
            self.condition = conditions[0]
        else:
            self.condition = conditions
            
        # Extract combustion conditions with default values
        self.fuel = self.condition.get('fuel')
        self.oxidizer_comp = self.condition.get('oxidizer')
        self.phi = float(self.condition.get('equivalence_ratio', 1.0))  # Default phi=1.0
        self.pres = float(self.condition.get('pressure', 100000.0))  # Default 1 atm (in Pa)
        self.temp = float(self.condition.get('temperature', 1800.0))  # Default 1800K
        
        # Mechanism file handling - defaults to GRI-Mech 3.0
        self.mech = self.config.get("mechanism", "gri30.yaml")
        self.output_dir = self.config.get("output_directory")
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists
        
        # Validate required parameters are present
        if not all([self.fuel, self.oxidizer_comp]):
            raise ValueError("Fuel and oxidizer composition must be specified in params.json")
        
        # Initialize Cantera gas phase object
        self.gas = ct.Solution(self.mech)
        self.n_reactions = self.gas.n_reactions
        self.simtype = 'HP'  # Constant pressure simulation type

    def calculate_ignition_delay(self, gas, plot=False):
        """
        Calculates the ignition delay time (IDT) for a given gas mixture.
        
        The IDT is determined as the time when the temperature gradient is maximum,
        indicating the onset of rapid heat release during ignition.
        
        Args:
            gas (ct.Solution): Cantera gas object with reaction mechanism
            plot (bool): If True, plots temperature profile (default False)
            
        Returns:
            float: Ignition delay time in seconds
        """
        # Set initial gas mixture composition and state
        gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer_comp)
        gas.TP = self.temp, self.pres
        
        # Calculate equilibrium temperature as stopping criterion
        gas.equilibrate(self.simtype)
        T_equi = gas.T
        
        # Reset to initial conditions for simulation
        gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer_comp)
        gas.TP = self.temp, self.pres
        
        # Set up constant pressure reactor
        r = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([r])
        
        # Set numerical tolerances for accurate integration
        sim.rtol = 1.0e-6  # Relative tolerance
        sim.atol = 1.0e-15  # Absolute tolerance

        # Simulation parameters
        t_end = 0.1  # Maximum simulation time (seconds)
        time = []
        temp = []
        states = ct.SolutionArray(gas, extra=['t'])  # For storing state history

        # Time integration loop
        while sim.time < t_end and r.T < T_equi:
            sim.step()
            time.append(sim.time)
            temp.append(r.T)
            states.append(r.thermo.state, t=sim.time)
        
        # Calculate ignition delay as point of maximum temperature gradient
        time = np.array(time)
        temp = np.array(temp)
        diff_temp = np.gradient(temp, time)  # Temperature derivative
        ign_pos = np.argmax(diff_temp)  # Index of ignition point
        
        return time[ign_pos]

    def compute_reaction_sensitivities(self):
        """
        Computes sensitivity coefficients showing how each reaction rate affects
        the ignition delay time.
        
        Sensitivity coefficients represent the derivative of temperature with
        respect to each reaction rate constant at the ignition point.
        
        Returns:
            tuple: (ignition_delay, sensitivity_coefficients)
                   - ignition_delay: Base case ignition delay (seconds)
                   - sensitivity_coefficients: Array of sensitivities for all reactions
        """
        # Initialize gas mixture
        self.gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer_comp)
        self.gas.TP = self.temp, self.pres
        
        # Set up reactor with sensitivity analysis enabled
        r = ct.IdealGasConstPressureReactor(self.gas)
        sim = ct.ReactorNet([r])
        
        # Enable sensitivity calculation for all reactions
        for i in range(self.n_reactions):
            r.add_sensitivity_reaction(i)
        
        # Set tolerances for both solution and sensitivities
        sim.rtol = 1.0e-6
        sim.atol = 1.0e-15 
        sim.rtol_sensitivity = 1.0e-6
        sim.atol_sensitivity = 1.0e-6

        # Get base case ignition delay first
        ign0 = self.calculate_ignition_delay(self.gas)
        
        # Run sensitivity simulation
        sens_data = []
        while sim.time < ign0 * 1.5:  # Integrate to 1.5x ignition time
            try:
                sim.step()
                # Store temperature sensitivities (3rd row in sensitivity matrix)
                sensitivities = sim.sensitivities()[2, :]  
                sens_data.append(sensitivities)
            except:
                break  # Terminate if simulation fails
        
        # Calculate average sensitivities around ignition point
        ign_pos = min(int(len(sens_data) * 0.8), len(sens_data)-1)  # Approximate ignition index
        sens_at_ignition = np.mean(sens_data[ign_pos-10:ign_pos+10], axis=0)  # Local average
        
        return ign0, sens_at_ignition

    def create_reduced_mechanism(self, threshold=0.1):
        """
        Creates a reduced chemical mechanism by removing the least sensitive reactions.
        
        The reduction is based on sensitivity analysis - reactions with the smallest
        impact on ignition delay are removed first.
        
        Args:
            threshold (float): Fraction of reactions to remove (0-1). Default 0.1 (10%).
            
        Returns:
            Path: Path to the saved reduced mechanism YAML file
            
        Raises:
            Exception: If verification of reduced mechanism fails
        """
        # Step 1: Compute reaction sensitivities
        ign0, sensitivities = self.compute_reaction_sensitivities()
        
        # Step 2: Identify least sensitive reactions to remove
        abs_sensitivities = np.abs(sensitivities)  # Consider magnitude only
        sorted_indices = np.argsort(abs_sensitivities)  # Sort reactions by sensitivity
        num_to_remove = int(self.n_reactions * threshold)
        reactions_to_remove = sorted_indices[:num_to_remove]  # Get least sensitive reactions
        
        # Step 3: Create new reaction list excluding removed reactions
        new_reactions = [r for i, r in enumerate(self.gas.reactions()) 
                        if i not in reactions_to_remove]
        
        # Step 4: Create new Cantera Solution with reduced mechanism
        reduced_gas = ct.Solution(
            thermo='IdealGas', 
            kinetics='GasKinetics',
            species=self.gas.species(),  # Keep all species
            reactions=new_reactions     # Only selected reactions
        )
        
        # Step 5: Save reduced mechanism to YAML file
        output_file = Path(self.output_dir) / f"reduced_{self.mech}"
        reduced_gas.write_yaml(output_file)
        
        print(f"Created reduced mechanism with {len(new_reactions)} reactions (reduced from {self.n_reactions})")
        print(f"Saved to: {output_file}")
        
        # Step 6: Validate reduced mechanism performance
        try:
            reduced_ign = self.calculate_ignition_delay(reduced_gas)
            error = abs(reduced_ign - ign0) / ign0 * 100
            print(f"Ignition delay change: {error:.2f}%")
        except Exception as e:
            print(f"Error verifying reduced mechanism: {str(e)}")
            raise
        
        return output_file

if __name__ == "__main__":
    # Command-line execution
    reducer = MechanismReducer()
    reduced_mech_file = reducer.create_reduced_mechanism(threshold=0.1)