import cantera as ct

import numpy as np
import json
import matplotlib.pyplot as plt
import time
import os
from scipy import integrate
from src.ChemicalMechanismGA.components.simulation_runner import SimulationRunner
from ..utils.save_species_conc import save_mole_fractions_to_json, save_species_concentrations
from src.ChemicalMechanismGA.components.constant_pressure_fitness import ConstantPressureFitness
from src.ChemicalMechanismGA.components.error_fitness import ConstantPressureError

class FitnessEvaluator:
    def __init__(self, mech, reactor_type, condition,
                 weights, difference_function, sharpening_factor, lam):
        """
        Initialize the FitnessEvaluator.

        Parameters:
            full_mech (str): Path to the original mechanism file.
            reactor_type (str): Type of reactor (e.g., "batch", "const_pressure").
            initial_temperature (float): Initial temperature for simulations.
            initial_pressure (float): Initial pressure for simulations.
            target_temperature (float): Target temperature for fitness evaluation.
            target_species (dict): Target mole fractions for key species (e.g., {"CO2": 0.1, "H2O": 0.2}).
            target_delay (float): Target ignition delay time.
            weight_temperature (float): Weight for temperature fitness.
            weight_species (float): Weight for species fitness.
            weight_ignition_delay (float): Weight for ignition delay fitness.
            difference_function (str): Method for calculating differences ("absolute", "squared", etc.).
            sharpening_factor (float): Sharpening factor for logarithmic/sigmoid difference functions.
        """
        self.mech = mech
        self.reactor_type = reactor_type
        self.condition = condition
        
        self.difference_function = difference_function
        self.sharpening_factor = sharpening_factor
        self.lam = lam
        self.weights = weights
        self.full_runner = SimulationRunner('gri30.yaml', self.reactor_type)
        
        # Create the fitness calculator based on reactor type
        if reactor_type == "constant_pressure":
            self.fitness_calculator = ConstantPressureError(self.difference_function, self.sharpening_factor, self.lam)
        else:
            raise ValueError(f"Unsupported reactor type: {reactor_type}")
        
        #self.target_delay = target_delay
        

    def create_reduced_mechanism(self, genome, write_to_file=False):
        """
        Create a reduced mechanism based on the genome.

        Parameters:
            genome (list): Binary genome representing active reactions.

        Returns:
            ct.Solution: Reduced mechanism as a Cantera Solution object.
        """
            
        gas = ct.Solution(self.mech)
        reactions = gas.reactions()
        reduced_reactions = [reaction for i, reaction in enumerate(reactions) if genome[i] == 1]
        #print(reduced_reactions)
        if len(reduced_reactions) < 50:  # Arbitrary threshold
            raise ValueError("Reduced mechanism has too few reactions")
        
        #This part of the code collects all species that are involved in the reduced mechanism's reactions.
        # Check for species usage
        species_used = set()
        for reaction in reduced_reactions:
            species_used.update(reaction.reactants.keys())
            species_used.update(reaction.products.keys())
        #print(f"Species used in reduced reactions: {species_used}")
        reduced_species = [sp for sp in gas.species() if sp.name in species_used]
        #print(f"Species used in reduced reactions: {reduced_species}")
        reduced_mech = ct.Solution(
            thermo="IdealGas",
            kinetics="GasKinetics",
            transport="mixture-averaged",
            species=reduced_species,
            reactions=reduced_reactions,
        )  
        
        # Ensure all reactions reference only species in the reduced mechanism
        for reaction in reduced_reactions:
            reaction_species = set(reaction.reactants.keys()).union(reaction.products.keys())
            for sp in reaction_species:
                if sp not in reduced_mech.species_names:
                    raise ValueError(f"Invalid mechanism: Reaction {reaction.equation} references missing species {sp}.")
       
        if write_to_file:
            file_path = f"reduced_mech_{len(reduced_reactions)}_rxns.yaml"
            reduced_mech.write_yaml(file_path)
            print(f"Reduced mechanism written to {file_path}")
        
        # Print summary of the reduced mechanism
        print(f"Reduced mechanism created with {len(reduced_reactions)} reactions and {len(reduced_mech.species_names)} species.")
        
        return reduced_mech
    

    
    def evaluate_fitness(self, genome, generation):
        """
        Evaluate the fitness of a genome by running a simulation with the reduced mechanism.

        Parameters:
            genome (list): Binary genome representing active reactions.
            generation (int): Current generation number (used for saving results).

        Returns:
            float: Fitness score (lower is better).
        """
        try:
            # Step 1: Create the reduced mechanism
            reduced_mech = self.create_reduced_mechanism(genome)
            
            try:
                T = self.condition['temperature']
                P = self.condition['pressure']
                X = {**self.condition['fuel'], **self.condition['oxidizer']}
                
                reduced_mech.TPX = T, P, X
                reduced_mech()               
            except Exception as e:
                print(f"Mechanism validation failed: {str(e)}")
                print(f"Failed condition: T={T}, P={P}, X={X}")
                return float('inf'), None

            runner = SimulationRunner(reduced_mech, self.reactor_type)
            reduced_results = runner.run_simulation(self.condition) 
            print("Run simulation was called succesfully")
            
            try:
                reduced_results["species_names"] = reduced_mech.species_names
                species_reduced = reduced_mech.species_names
                print(len(species_reduced), ": " , species_reduced)
                
                mole_fractions = {species: reduced_results["mole_fractions"][i] for i, species in enumerate(reduced_results["species_names"])}
                reduced_results["mole_fractions"] = mole_fractions
            except Exception as e:
                print(f"Error in evaluating reduced mechanism species: {e}")
            
            # Step 4: Run simulation with full mechanism
            
            full_results = self.full_runner.run_simulation(self.condition)
            key_species = ['CH4', 'O2', 'CO2', 'H2O', 'CO', 'H2', 'O', 'OH', 'H', 'CH3']
            
            # Step 4: Calculate fitness
            print("About to call fitness calcualtion")
            epsilon=0
            fitness = self.fitness_calculator.combined_fitness(
                reduced_results,
                full_results,
                key_species,
                self.weights
            )
            
            print(f"Fitness Score for Generation {generation}: {fitness}")
            return fitness, reduced_results
        
        except Exception as e:
            print(f"Error during fitness evaluation for genome : {e}") #!!!!!!!!!!!!!!!!!!
            return 1e6, None  # Penalize invalid solutions

    def run_generation(self, population, generation, save_top_n):
        """
        Run a single generation of the genetic algorithm.

        Parameters:
            population (list): List of genomes in the current generation.
            generation (int): Current generation number.
            filename (str): Filename for saving mole fractions.
            species_filename (str): Filename for saving species concentrations.

        Returns:
            list: Fitness scores of the current generation.
        """
            # File paths for saving results
        base_dir = f"results/generation_{generation}"
        os.makedirs(base_dir, exist_ok=True)

        # Initialize tracking variables
        fitness_scores = []
        all_results = []
        reaction_counts = []
        species_usage = {}
        # Access the individuals array from the Population object
        individuals = population.individuals # gets array of population
        print(f"Processing {len(individuals)} individuals in generation {generation}")

        for i, genome in enumerate(individuals): #goes through all the genomes(individuals)
            fitness, results = self.evaluate_fitness(genome, generation)
            
            fitness_scores.append(fitness)
            print(f"Genome Number: {len(fitness_scores)}")
            
            if results is not None:
                #count active reactions
                reaction_count = sum(genome)
                reaction_counts.append(reaction_count)
                print(f"Length of reaction_counts: {len(reaction_counts)}")
                
                max_temp = results.get("max_temperature", 0)
             
                result_entry= {
                    "fitness": fitness,
                    "reaction_count": reaction_count,
                    "max_temperature": max_temp,
                    "genome": genome.copy() if hasattr(genome, "copy") else list(genome),
                    "individual_index": i
                }
                all_results.append(result_entry)
                
        # Sort results bby fitness
        all_results.sort(key=lambda x: x["fitness"])   
        
        # Calculate statistics
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else float('inf')
        min_fitness = min(fitness_scores) if fitness_scores else float('inf')
        max_fitness = max(fitness_scores) if fitness_scores else float('inf')
        std_fitness = np.std(fitness_scores) if len(fitness_scores) > 1 else 0

        avg_reactions = sum(reaction_counts) / len(reaction_counts) if reaction_counts else 0
        min_reactions = min(reaction_counts) if reaction_counts else 0
        max_reactions = max(reaction_counts) if reaction_counts else 0      
        
        # Save detailed results for top performers
        top_n_results = all_results[:save_top_n]
        #print(f"top n results: {top_n_results}")

        for rank, result in enumerate(top_n_results):
            individual_idx = result["individual_index"]
            detailed_results = result  # Use already computed results

            # Save mole fractions and other data
            if detailed_results is not None and "mole_fractions" in detailed_results:
                species_names = detailed_results.get("species_names", [])

                # Save to JSON
                filename = f"{base_dir}/rank_{rank+1}_individual_{individual_idx}.json"
                save_mole_fractions_to_json(detailed_results, species_names, generation, filename)
                
                # Save mechanism details
                with open(f"{base_dir}/rank_{rank+1}_mechanism_info.txt", 'w') as f:
                    f.write(f"Fitness: {result['fitness']}\n")
                    f.write(f"Reaction count: {result['reaction_count']}\n")
                    f.write(f"Individual index: {individual_idx}\n")

                    # Save active reaction indices
                    active_reactions = [i for i, active in enumerate(genome) if active]
                    f.write(f"Active reaction indices: {active_reactions}\n")
                    
   
        
        # Convert numpy types to Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            elif isinstance(obj, (np.int64, np.int32, np.int16)):
                return int(obj)  # Convert numpy integers to Python int
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)  # Convert numpy floats to Python float
            else:
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # # Save generation statistics
        # with open(f"{base_dir}/generation_stats.json", 'w') as f:
        #     json.dump(generation_stats, f, indent=2, default=convert_numpy_types)
            
            # Plot fitness distribution
        # if len(fitness_scores) > 1:
        #     plt.figure(figsize=(10, 6))
        #     plt.hist(fitness_scores, bins=20, alpha=0.7)
        #     plt.title(f"Fitness Distribution - Generation {generation}")
        #     plt.xlabel("Fitness Score")
        #     plt.ylabel("Count")
        #     plt.savefig(f"{base_dir}/fitness_distribution.png")
        #     plt.close()

          
        return {
            "fitness_scores": fitness_scores,
            "best_genome": all_results[0]["genome"] if all_results else None,
            "best_fitness": min_fitness,
            "active_reactions": min_reactions,
            "average_reactions": avg_reactions
    }
                    

    def calculate_premix_fitness(self, species_reduced, reduced_results, epsilon):
        total_error = 0.0
        
        # Create SimulationRunner instances for full and reduced mechanisms
        full_runner = SimulationRunner('gri30.yaml', self.reactor_type)
            
        # Run simulations for this condition
        full_profiles = full_runner.run_simulation(self.condition)
       
        reduced_profiles = reduced_results
        #print(f"Full profiles for condition : {full_profiles}")
        #print(f"Reduced profiles for condition : {reduced_profiles}")

        if not isinstance(reduced_profiles, dict):
            raise ValueError(f"Expected reduced_profiles to be a dictionary, got {type(reduced_profiles)}")
        if not isinstance(full_profiles, dict):
            raise ValueError(f"Expected full_profiles to be a dictionary, got {type(full_profiles)}")
        
        z = full_profiles['grid'] 
        #is_uniform_grid = np.allclose(np.diff(z), np.diff(z).mean())
        
        condition_error = 0.0
        for k, species in enumerate(species_reduced):
            if species not in full_profiles or species not in reduced_profiles["mole_fractions"]:
                continue
            #print(f"Processing species: {species}")
            Y_orig = full_profiles[species]
            Y_calcd = reduced_profiles["mole_fractions"][species]
            
            # print(f"Species: {species}")
            # print(f"Y_orig: {Y_orig}")
            # print(f"Y_calcd: {Y_calcd}")
            
            # W_k = 1.0 if np.max(Y_orig) >= 1e-7 else 0.0
            # if W_k == 0.0:
            #     continue
            W_k = 1.0
            l2_orig = np.sqrt(np.trapezoid(Y_orig**2, z))
            
            diff = Y_calcd - Y_orig
            l2_diff = np.sqrt(np.trapezoid(diff**2, z))
            
            if l2_orig > 0:
                relative_error = W_k * l2_diff / l2_orig
            else:
                relative_error = 0 if l2_diff == 0 else W_k # handle edge cases
                
            # print(f"L2_orig: {l2_orig}, L2_diff: {l2_diff}")
            # print(f"Relative error for {species}: {relative_error}")
                
            condition_error += relative_error
            # print(f"Condition error: {condition_error}\n")
        
        total_error += condition_error
                
        return 1.0/(epsilon + total_error)


    def calculate_psr_fitness(self, species_reduced, reduced_results, epsilon):
            total_error = 0.0
            
            # Create SimulationRunner instances for full and reduced mechanisms
            full_runner = SimulationRunner('gri30.yaml', self.reactor_type)
                
            # Run simulations for this condition
            full_profiles = full_runner.run_simulation(self.condition)
        
            reduced_profiles = reduced_results
            #print(f"Full profiles for condition : {full_profiles}")
            #print(f"Reduced profiles for condition : {reduced_profiles}")
            
            if not isinstance(reduced_profiles, dict):
                raise ValueError(f"Expected reduced_profiles to be a dictionary, got {type(reduced_profiles)}")
            if not isinstance(full_profiles, dict):
                raise ValueError(f"Expected full_profiles to be a dictionary, got {type(full_profiles)}")
            
            full_time = full_profiles['time'] # change to grid for PREMIX
            print(f"Full time shape: {full_time.shape}, type: {type(full_time)}")
            #is_uniform_grid = np.allclose(np.diff(z), np.diff(z).mean())
            reduced_time = reduced_profiles['time']  # Time points for the reduced mechanism
            print(f"Reduced time shape: {reduced_time.shape}, type: {type(reduced_time)}")
            
            # Truncate the reduced profile to match the full profile's time range
            max_time = full_time[-1]  # Last time point of the full profile
            valid_indices = reduced_time <= max_time  # Indices where reduced time is within the full time range
            truncated_reduced_time = reduced_time[valid_indices]  # Truncated reduced time array
            print(f"Truncated reduced time shape: {truncated_reduced_time.shape}")
            
            # Truncate mole fractions for reduced profiles
            truncated_reduced_profiles = {}
            for species in reduced_profiles["mole_fractions"]:
                truncated_reduced_profiles[species] = reduced_profiles["mole_fractions"][species][valid_indices]
            
            # Interpolate the full profile to match the truncated reduced time array
            interpolated_full_profiles = {}
            for species in full_profiles:
                if species == "time":  # Skip the time array
                    continue
                interpolated_full_profiles[species] = np.interp(
                    truncated_reduced_time, full_time, full_profiles[species]
                )
                
            condition_error = 0.0
            for k, species in enumerate(species_reduced):
                if species not in interpolated_full_profiles or species not in truncated_reduced_profiles:
                    continue
                #print(f"Processing species: {species}")
                Y_orig = interpolated_full_profiles[species]
                
                Y_calcd = truncated_reduced_profiles[species]
                print(f"Y_orig shape: {Y_orig.shape}, Y_calcd shape: {Y_calcd.shape}")
                
                # print(f"Species: {species}")
                # print(f"Y_orig: {Y_orig}")
                # print(f"Y_calcd: {Y_calcd}")
                
                # W_k = 1.0 if np.max(Y_orig) >= 1e-7 else 0.0
                # if W_k == 0.0:
                #     continue
                W_k = 1.0
                l2_orig = np.sqrt(np.trapezoid(Y_orig**2, truncated_reduced_time))
                
                diff = Y_calcd - Y_orig
                l2_diff = np.sqrt(np.trapezoid(diff**2, truncated_reduced_time))
                
                if l2_orig > 0:
                    relative_error = W_k * l2_diff / l2_orig
                else:
                    relative_error = 0 if l2_diff == 0 else W_k # handle edge cases
                    
                # print(f"L2_orig: {l2_orig}, L2_diff: {l2_diff}")
                # print(f"Relative error for {species}: {relative_error}")
                    
                condition_error += relative_error
                # print(f"Condition error: {condition_error}\n")
            
            total_error += condition_error
                    
            return 1.0/(epsilon + total_error)
# ALL previous fitness functions are defined below
