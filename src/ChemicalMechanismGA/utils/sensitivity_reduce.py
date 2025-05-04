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
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.key_species = self.config.get("key_species", [])
        
        # Handle both list and dictionary format for conditions
        conditions = self.config.get("conditions", {})
        if isinstance(conditions, list):
            # Take the first condition if it's a list
            self.condition = conditions[0]
        else:
            self.condition = conditions
            
        self.fuel = self.condition.get('fuel')
        self.oxidizer_comp = self.condition.get('oxidizer')
        self.phi = float(self.condition.get('equivalence_ratio', 1.0))
        self.pres = float(self.condition.get('pressure', 100000.0))
        self.temp = float(self.condition.get('temperature', 1800.0))
        self.mech = self.config.get("mechanism", "gri30.yaml")
        self.output_dir = self.config.get("output_directory")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not all([self.fuel, self.oxidizer_comp]):
            raise ValueError("Fuel and oxidizer composition must be specified in params.json")
        
        self.gas = ct.Solution(self.mech)
        self.n_reactions = self.gas.n_reactions
        self.simtype = 'HP'
        
    def calculate_ignition_delay(self, gas, plot=False):
        """Calculate ignition delay time for given gas object"""
        gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer_comp)
        gas.TP = self.temp, self.pres
        
        # Get equilibrium temperature for ignition break
        gas.equilibrate(self.simtype)
        T_equi = gas.T
        
        gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer_comp)
        gas.TP = self.temp, self.pres
        # Reactor setup
        r = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([r])
        sim.rtol = 1.0e-6
        sim.atol = 1.0e-15

        t_end = 0.1  # seconds
        time = []
        temp = []
        states = ct.SolutionArray(gas, extra=['t'])

        while sim.time < t_end and r.T < T_equi:
            sim.step()
            time.append(sim.time)
            temp.append(r.T)
            states.append(r.thermo.state, t=sim.time)
        
        time = np.array(time)
        temp = np.array(temp)
        diff_temp = np.gradient(temp, time)
        ign_pos = np.argmax(diff_temp)
        return time[ign_pos]

    def compute_reaction_sensitivities(self):
        """Compute sensitivity coefficients for all reactions"""
        self.gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer_comp)
        self.gas.TP = self.temp, self.pres
        
        # Reactor setup with sensitivity
        r = ct.IdealGasConstPressureReactor(self.gas)
        sim = ct.ReactorNet([r])
        
        # Enable sensitivity for all reactions
        for i in range(self.n_reactions):
            r.add_sensitivity_reaction(i)
        
        sim.rtol = 1.0e-6
        sim.atol = 1.0e-15
        sim.rtol_sensitivity = 1.0e-6
        sim.atol_sensitivity = 1.0e-6

        # Calculate ignition delay first
        ign0 = self.calculate_ignition_delay(self.gas)
        
        # Run sensitivity simulation
        sens_data = []
        while sim.time < ign0 * 1.5:
            try:
                sim.step()
                sensitivities = sim.sensitivities()[2, :]  # Temperature sensitivity
                sens_data.append(sensitivities)
            except:
                break
        
        # Get sensitivities at ignition point
        ign_pos = min(int(len(sens_data) * 0.8), len(sens_data)-1)  # Approximate ignition position
        sens_at_ignition = np.mean(sens_data[ign_pos-10:ign_pos+10], axis=0)
        
        return ign0, sens_at_ignition

    def create_reduced_mechanism(self, threshold=0.1):
        """
        Create reduced mechanism by removing least sensitive reactions
        threshold: fraction of reactions to remove (0.1 = 10%)
        """
        # Compute sensitivities
        ign0, sensitivities = self.compute_reaction_sensitivities()
        
        # Identify least sensitive reactions
        abs_sensitivities = np.abs(sensitivities)
        sorted_indices = np.argsort(abs_sensitivities)
        num_to_remove = int(self.n_reactions * threshold)
        reactions_to_remove = sorted_indices[:num_to_remove]
        
        # Create new reaction list excluding least sensitive reactions
        new_reactions = [r for i, r in enumerate(self.gas.reactions()) 
                        if i not in reactions_to_remove]
        
        # Create new gas object with reduced mechanism
        reduced_gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                                species=self.gas.species(), reactions=new_reactions)
        
        # Save to YAML file
        output_file = Path(self.output_dir) / f"reduced_{self.mech}"
        reduced_gas.write_yaml(output_file)
        
        print(f"Created reduced mechanism with {len(new_reactions)} reactions")
        print(f"Saved to: {output_file}")
        
        # Verify the reduced mechanism
        try:
            reduced_ign = self.calculate_ignition_delay(reduced_gas)
            error = abs(reduced_ign - ign0) / ign0 * 100
            # print(f"Original IDT: {ign0*1000:.2f} ms")
            # print(f"Reduced IDT: {reduced_ign*1000:.2f} ms")
            print(f"Error: {error:.2f}%")
        except Exception as e:
            print(f"Error verifying reduced mechanism: {str(e)}")
        
        return output_file

if __name__ == "__main__":
    reducer = MechanismReducer()
    reduced_mech_file = reducer.create_reduced_mechanism(threshold=0.1)