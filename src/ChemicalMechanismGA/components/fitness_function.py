import cantera as ct
import math
import numpy as np
import json
import matplotlib.pyplot as plt
import time
import os
from scipy import integrate
from src.ChemicalMechanismGA.components.simulation_runner import SimulationRunner
from ..utils.save_species_conc import save_mole_fractions_to_json, save_species_concentrations

class FitnessEvaluator:
    def __init__(self, mech, reactor_type, condition,
                 weight_species, difference_function, sharpening_factor):
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
        #self.target_temperature = target_temperature
        
        self.difference_function = difference_function
        self.sharpening_factor = sharpening_factor
       
        self.weight_species = weight_species
        #self.weight_ignition_delay = weight_ignition_delay
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
            

            # Step 2: Run the simulation
            #reduced_results = self.run_reduced_simulation(reduced_mech) # works till here
            
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
            
            # Step 4: Calculate fitness
            #print("About to call premix fitness calcualtion")
            epsilon=0
            fitness = self.calculate_premix_fitness(species_reduced, reduced_results, epsilon)
            
            print(f"Fitness Score for Generation {generation}: {fitness}")
            return fitness, reduced_results
        
        except Exception as e:
            print(f"Error during fitness evaluation for genome : {e}")
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
        
        start_time = time.time()

        for i, genome in enumerate(individuals): #goes through all the genomes(individuals)
            fitness, results = self.evaluate_fitness(genome, generation)
            
            fitness_scores.append(fitness)
            print(f"Length of fitness_scores: {len(fitness_scores)}")
            
            if results is not None:
                #count active reactions
                reaction_count = sum(genome)
                reaction_counts.append(reaction_count)
                print(f"Length of reaction_counts: {len(reaction_counts)}")
                
                max_temp = results.get("max_temperature", 0)
             
                # # Track which species are used in the best mechanism
                # if i < save_top_n:
                #     for species in results.get("species_names", []):
                #         species_usage[species] = species_usage.get(species, 0) + 1
                        
                #Store full results for top performers
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
                    
            # Generate summary statistics
        generation_stats = {
            "generation": generation,
            "time_taken": time.time() - start_time,
            "fitness": {
                "min": min_fitness,
                "max": max_fitness,
                "avg": avg_fitness,
                "std": std_fitness
            },
            "reactions": {
                "min": min_reactions,
                "max": max_reactions,
                "avg": avg_reactions
            },
            "top_performers": all_results[:save_top_n],
            "most_common_species": sorted(species_usage.items(), key=lambda x: x[1], reverse=True)[:20]
        }
        
        
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
        
        # Save generation statistics
        with open(f"{base_dir}/generation_stats.json", 'w') as f:
            json.dump(generation_stats, f, indent=2, default=convert_numpy_types)
            
            # Plot fitness distribution
        if len(fitness_scores) > 1:
            plt.figure(figsize=(10, 6))
            plt.hist(fitness_scores, bins=20, alpha=0.7)
            plt.title(f"Fitness Distribution - Generation {generation}")
            plt.xlabel("Fitness Score")
            plt.ylabel("Count")
            plt.savefig(f"{base_dir}/fitness_distribution.png")
            plt.close()

            # # Plot fitness vs reaction count
            # plt.figure(figsize=(10, 6))
            # plt.scatter(reaction_counts, fitness_scores, alpha=0.5)
            # plt.title(f"Fitness vs Reaction Count - Generation {generation}")
            # plt.xlabel("Number of Reactions")
            # plt.ylabel("Fitness Score")
            # plt.savefig(f"{base_dir}/fitness_vs_reactions.png")
            # plt.show()
            # plt.close()

        # Print summary
        print(f"\nGeneration {generation} completed in {time.time() - start_time:.2f} seconds")
        print(f"Best fitness: {min_fitness:.6f}, Avg fitness: {avg_fitness:.6f}")
        print(f"Reaction counts - Min: {min_reactions}, Avg: {avg_reactions:.1f}, Max: {max_reactions}")

        return {
            "fitness_scores": fitness_scores,
            "best_genome": all_results[0]["genome"] if all_results else None,
            "best_fitness": min_fitness
    }
                    

    def calculate_difference(self, actual, target):
        if self.difference_function == "absolute":
            return abs(actual - target)
        elif self.difference_function == "squared":
            return (actual - target) ** 2
        elif self.difference_function == "relative":
            return abs((actual - target) / target) if target != 0 else float("inf")
        elif self.difference_function == "logarithmic":
            return math.log(1 + self.sharpening_factor * abs((actual - target) / target)) if target != 0 else float("inf")
        elif self.difference_function == "sigmoid":
            return 1 / (1 + math.exp(self.sharpening_factor * (1 - actual / target))) if target != 0 else float("inf")
        else:
            raise ValueError(f"Unsupported difference function: {self.difference_function}")

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

# ALL previous fitness functions are defined below
#region
    def temperature_fitness(self, results):
        """
        Calculate fitness based on the difference between the actual and target temperature.

        Parameters:
            results (dict): Simulation results containing temperature.

        Returns:
            float: Fitness score (lower is better).
        """
        try:
            actual_temperature = results.get("temperature", 0.0)
            fitness = self.calculate_difference(actual_temperature, self.target_temperature)
            print(f"Temperature Fitness: {fitness} (Actual: {actual_temperature}, Target: {self.target_temperature})")
            return fitness
        except Exception as e:
            print(f"Error during temperature fitness calculation: {e}")
            return None

    def species_fitness(self, results):
        """
        Calculate fitness based on the difference between actual and target species mole fractions.

        Parameters:
            results (dict): Simulation results containing mole fractions.

        Returns:
            float: Fitness score (lower is better).
        """
        try:
            mole_fractions = results.get("mole_fractions", None)
            #print(type(mole_fractions))
            if mole_fractions is None:
                raise ValueError("Mole fractions missing in the results")
            
            species_name_to_index = {name: i for i, name in enumerate(results["species_names"])}
            #print("Species Name to index:", species_name_to_index)
            #print(type(species_name_to_index))
            fitness = 0.0
            #print("Target species items as passed: ", self.target_species.items()) # passed as a dict
            for species, target_fraction in self.target_species.items():
                if species not in species_name_to_index:
                    print(f"Warning! species {species} not found in the mechanism.")
                    actual_fraction = 0.0
                else:
                    #actual_fraction = mole_fractions[species_name_to_index[species]]
                    actual_fraction = mole_fractions.get(species, 0.0)
                    print(f"The actual fraction of species: {species} ", actual_fraction)
                    
                fitness += self.calculate_difference(actual_fraction, target_fraction)
                print(f"Species Fitness for {species}: {self.calculate_difference(
                    actual_fraction, target_fraction)} (Actual: {actual_fraction}, Target: {target_fraction})")
            fitness /= len(self.target_species)
            return fitness
        except Exception as e:
            print(f"Error during species fitness calculation: {e}")
            return None

    
    def ignition_delay_fitness(self, results):
        """
    Calculate fitness based on the difference between actual and target ignition delay time.

    Parameters:
        results (dict): Simulation results containing ignition delay time.
        target_delay (float): Target ignition delay time (in seconds).

    Returns:
        float: Fitness score (lower is better).
    """
        try:
            actual_delay = results.get("ignition_delay", 0.0)
            fitness = self.calculate_difference(actual_delay, self.target_delay)
            print(f"Ignition Delay Fitness: {fitness} (Actual: {actual_delay}, Target: {self.target_delay})")
            return fitness
        except Exception as e:
            print(f"Error during ignition delay fitness calculation: {e}")
            return None
    
    def combined_fitness(self, results):
        """
        Combine temperature and species fitness into a single score.

        Parameters:
            results (dict): Simulation results containing temperature and mole fractions.

        Returns:
            float: Combined fitness score (lower is better).
        """
        try:
            temp_fitness = self.temperature_fitness(results) * self.weight_temperature
            species_fitness = self.species_fitness(results) * self.weight_species
            ignition_fitness = self.ignition_delay_fitness(results) * self.weight_ignition_delay
        
            total_fitness = temp_fitness + species_fitness + ignition_fitness
            print(f"Combined Fitness: {total_fitness}")
            return total_fitness
        except Exception as e:
            print(f"Error during combined_fitness calculation: {e}")
            return None
#endregion