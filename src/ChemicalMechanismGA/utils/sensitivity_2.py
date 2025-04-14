
import sys
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import timeit
from ..components.fitness_function import FitnessEvaluator

# Todo: compute average sensitivity for different cases.
# add more conditons in condition dict.
# or create a separe condition file and read from that(better approach)

class SensitivityAnalysis:
    def __init__(self, mech, reactor_type, condition):
        self.mech = mech
        self.reactor_type = reactor_type
        self.condition = condition
        self.path = 'outputs'
        self.gas = ct.Solution(self.mech)
        self.n_reactions = len(self.gas.reactions())
        
        try:
            self.T = self.condition['temperature']
            self.P = self.condition['pressure']
            self.X = {**self.condition['fuel'], **self.condition['oxidizer']}
            self.gas.TPX = self.T, self.P, self.X
                            
        except Exception as e:
            print(f"Mechanism validation failed: {str(e)}")
            print(f"Failed condition: T={self.T}, P={self.P}, X={self.X}")
            return float('inf'), None
    
    def ign_uq(self, factor, bootplot=True):
    # This function calculates the ignition delay time (IDT) 
    # for a given set of reaction rate multipliers
    # Calculates the ignition delay time (IDT), which is the time at which 
    # the system reaches the point of maximum temperature gradient (rapid heat release).
    
        self.gas.set_multiplier(1.0) # reset all multipliers
        for i in range(self.gas.n_reactions):
            self.gas.set_multiplier(factor[i],i)
            #print(gas.reaction_equations(i)+' index_reaction:',i,'multi_factor:',factor[i])
        
        # self.gas.set_equivalence_ratio( phi, fuel, 'O2:1.0, N2:7.52' )
        # self.gas.TP = temp, pres*ct.one_atm
        print(f"condition: T={self.T}, P={self.P}, X={self.X}")
        
        # here it is constant pressure with IdealGasReactor
        r = ct.IdealGasConstPressureReactor(self.gas)
        sim = ct.ReactorNet([r])
        
        # set the tolerances for the solution and for the sensitivity coefficients
        sim.rtol = 1.0e-6
        sim.atol = 1.0e-15

        t_end = 0.02 # time in seconds
        time = []
        temp = []
        states = ct.SolutionArray(self.gas, extra=['t'])

        # stateArray = []
        while sim.time < t_end: #and r.T < T_equi - 0.1
            sim.step()
            time.append(sim.time)
            temp.append(r.T)
            states.append(r.thermo.state, t=sim.time)
            
        time = np.array(time)
        temp = np.array(temp)
        diff_temp = np.diff(temp)/np.diff(time)
        if bootplot:
            # print(r.T, T_equi)
            plt.figure()
            plt.plot(states.t, states.T, '-ok')
            # plt.plot( 1.0, states.T[ign_pos], 'rs',markersize=6  )
            plt.xlabel('Time [s]')
            plt.ylabel('T [K]')
            #plt.xlim( 0.99, 1.01 )
            #plt.ylim( -0.001,0.001 )
            plt.tight_layout()
            plt.show()
        ign_pos = np.argmax( diff_temp )
        ign = time[ign_pos]
        self.gas.set_multiplier(1.0) # reset all multipliers

        return ign #Returns the ignition delay time (IDT) as a single value
    
    def compute_sensitivity(self):
        
        #average_sensitivities = np.zeros((self.n_reactions,))
        rxn_OH_sensitivity = []
        
        if self.reactor_type == "constant_pressure":
            self.reactor = ct.IdealGasConstPressureReactor(self.gas)
            self.sim = ct.ReactorNet([self.reactor])
            
            # Enable sensitivity analysis for all reactions
        for i in range(self.n_reactions):
            self.reactor.add_sensitivity_reaction(i)
            
        for i in range(self.n_reactions):
            rxn_OH_sensitivity.append([])
        
        factor = np.ones(self.gas.n_reactions ) #?????
        ign0 = self.ign_uq(factor, False)
        print("Ignition Delay is: {:.4f} ms".format(ign0*1000))
        while self.sim.time < ign0 * 1.5:
            try:
                self.sim.step()
            except Exception as e:
                print("It is a CanteraError during combustion simulation for sensitivity analysis! This computation ends and potentially gives incomplete sensitivity curve.")
                break
            except RuntimeError:
                print("It is a RuntimeError during combustion simulation for sensitivity analysis! This computation ends and potentially gives incomplete sensitivity curve.")
                break
            #self.time.append(self.sim.time)
            
            for i in range(self.n_reactions):
                s = self.sim.sensitivity('OH', i)
                rxn_OH_sensitivity[i].append(s)
            
        rxn_OH_sensitivity_max = [np.max(np.abs(np.array(s))) for s in rxn_OH_sensitivity]
        
        # Print results
        # for i, sens in enumerate(rxn_OH_sensitivity_max):
        #     print(f"Reaction {i}: Max Sensitivity = {sens:.4e}")

        return ign0, rxn_OH_sensitivity_max
        
    def reduce_with_sensitivity(self, data_s):
        try:
            # Load the CSV file and extract the second column
            rxn_sensitivity_OH = pd.read_csv(data_s, index_col=0, skiprows=[1])  # Use the first column as the index
            sensitivity_values = rxn_sensitivity_OH.iloc[:, 0].values  # Extract the second column as a NumPy array
            print("File loaded successfully.")
        except FileNotFoundError:
            print("File not found in path")
            raise FileNotFoundError

        # Sort sensitivities in ascending order
        sorted_indices = np.argsort(sensitivity_values)

        # Create a genome with all reactions active (1)
        genome = np.ones(self.n_reactions, dtype=int)

        # Take 10% of the least sensitive reactions and deactivate them (set to 0)
        num_to_deactivate = int(0.1 * self.n_reactions)
        least_sensitive_indices = sorted_indices[:num_to_deactivate]
        genome[least_sensitive_indices] = 0

        print(f"Genome after deactivating 10% least sensitive reactions: {genome}")
        return genome
        

def main():
    mechanism = 'gri30.yaml'
    reactor_type = 'constant_pressure'
    condition = {
            'phi': 1.0,
            'fuel': {'CH4':1.0}, #mole fraction
            'oxidizer': {'O2': 1.0, 'N2': 7.52},
            'pressure': 2670 * ct.one_atm / 101325, 
            'temperature': 1800.0,
    }

    # Create the sensitivity analysis object
    sensitivity_analysis = SensitivityAnalysis(mechanism, reactor_type, condition)
    # Perform sensitivity analysis
    IDT_full, sensitivity = sensitivity_analysis.compute_sensitivity()
    data = pd.DataFrame(sensitivity)
    data.to_csv("outputs" + "/CH4_sensitivity.csv")
    print("Sensitivity file saved.")
    
    data_s = "outputs/CH4_sensitivity.csv"    
    sens_reduced_genome = sensitivity_analysis.reduce_with_sensitivity(data_s)
    
    weights = {"temperature": 1, "species": 1, "ignition_delay": 1}
    difference_function = "logarithmic"  # Default difference function
    sharpening_factor = 6 # empirical value
    lam = 0.1
   
    fitness_evaluator = FitnessEvaluator(
        mechanism,
        reactor_type,
        condition,
        weights,
        difference_function,
        sharpening_factor,
        lam
        )
    
    reduced_mech, reduced_file_path = fitness_evaluator.create_reduced_mechanism(sens_reduced_genome, True)
    
    # Calculate IDT for the full mechanism
    print("\nCalculating IDT for the full mechanism...")
    full_mech_idt = IDT_full
    print("Full Mechanism IDT: {:.4f} ms".format(full_mech_idt*1000)) 

    # Calculate IDT for the reduced mechanism
    print("\nCalculating IDT for the reduced mechanism...")
    reduced_sensitivity_analysis = SensitivityAnalysis(reduced_file_path, reactor_type, condition)
    reduced_mech_idt = reduced_sensitivity_analysis.ign_uq(factor=np.ones(reduced_sensitivity_analysis.n_reactions), bootplot=False)
    print("Reduced Mechanism IDT: {:.4f} ms".format(reduced_mech_idt*1000)) 
    
    # Compare the IDTs
    idt_difference = abs(full_mech_idt - reduced_mech_idt)
    print("\nDifference in IDT between full and reduced mechanisms: {:.4f} ms".format(idt_difference*1000)) 
    
    percentage_error = (idt_difference/full_mech_idt) *100
    print("\nThe percentage error: ", percentage_error)
    
if __name__ == "__main__":
    main()
    