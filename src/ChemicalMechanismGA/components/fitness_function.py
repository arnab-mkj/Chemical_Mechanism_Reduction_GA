import cantera as ct

import numpy as np
import json
import matplotlib.pyplot as plt
import time
import os
from scipy import integrate
from src.ChemicalMechanismGA.components.simulation_runner import SimulationRunner
from ..utils.save_species_conc import save_mole_fractions_to_json, save_species_concentrations
# from src.ChemicalMechanismGA.components.constant_pressure_fitness import ConstantPressureFitness
from src.ChemicalMechanismGA.components.error_fitness import ConstantPressureFitness
from src.ChemicalMechanismGA.components.IDT_calc import idt_value
import math

class FitnessEvaluator:
    def __init__(self, mech, reactor_type, conditions,
                 weights, genome_length,  difference_function, 
                 sharpening_factor, normalization_method, key_species):
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
        self.conditions = conditions
        self.genome_length = genome_length
        
        self.dfunc = difference_function
        self.norm_func = normalization_method
        self.sharpening_factor = sharpening_factor
        self.weights = weights
        self.full_mech = SimulationRunner(self.mech, self.reactor_type)
        self.key_species = key_species
        
        # Create the fitness calculator based on reactor type
        if reactor_type == "constant_pressure":
            self.fitness_calculator = ConstantPressureFitness(self.dfunc, self.sharpening_factor, self.norm_func, self.conditions, self.weights)
        else:
            raise ValueError(f"Unsupported reactor type: {reactor_type}")
        
        #self.target_delay = target_delay
        

    def create_reduced_mechanism(self, genome, write_to_file, keep_all_species, condition): # to false
            
        gas = ct.Solution(self.mech)
        reactions = gas.reactions() #gets reaactions from full mech
        
        #print(reduced_reactions)
        # Identify duplicate reactions
        duplicate_groups = {}
        for i, reaction in enumerate(reactions):
            if hasattr(reaction, 'duplicate') and reaction.duplicate:
                key = tuple(sorted(reaction.reactants.items()))  # Use reactants as a key
                if key not in duplicate_groups:
                    duplicate_groups[key] = []
                duplicate_groups[key].append(i)
                
        # Ensure all duplicates are included or excluded together
        for group in duplicate_groups.values():
            if any(genome[i] == 1 for i in group):
                for i in group:
                    genome[i] = 1  # Include all duplicates in the group
            else:
                for i in group:
                    genome[i] = 0  # Exclude all duplicates in the group
            
            
        reduced_reactions = [reaction for i, reaction in enumerate(reactions) if genome[i] == 1]
        if len(reduced_reactions) < 50:  # Arbitrary threshold
                raise ValueError("Reduced mechanism has too few reactions")
        #This part of the code collects all species that are involved in the reduced mechanism's reactions.
        # Check for species usage
        
        if keep_all_species:
            reduced_species = gas.species()
            print("Keeping all species intact in the reduced mechanism")
        else:
            species_used = set()
            for reaction in reduced_reactions:
                species_used.update(reaction.reactants.keys())
                species_used.update(reaction.products.keys())
                
            fuel_species = condition['fuel'].keys()
            oxidizer_species = condition['oxidizer'].keys()
            species_used.update(fuel_species)
            species_used.update(oxidizer_species)
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
        else:
            file_path = " "
        # Print summary of the reduced mechanism
        print(f"Reduced mechanism created with {len(reduced_reactions)} reactions and {len(reduced_mech.species_names)} species.")
        
        return reduced_mech, file_path
    

    
    def evaluate_fitness(self, genome, generation):
      
        try:
            # Step 1: Create the reduced mechanism
            
            total_fitness = {
                "combined_fitness": 0.0,
                "temperature_fitness": 0.0,
                "species_fitness": 0.0,
                "ignition_delay_fitness": 0.0,
                "reaction_count_fitness": 0.0
            }
            num_conditions = len(self.conditions)
            reduced_results ={}
            
            for idx, condition in enumerate (self.conditions):
                try:
                    print(f"Type of condition: {type(condition)}, Value: {condition}")
                    
                    reduced_mech, reduced_file_path = self.create_reduced_mechanism(
                                                        genome, 
                                                        write_to_file=False,
                                                        keep_all_species=False, 
                                                        condition =condition)
                    
                    T = condition['temperature']
                    P = condition['pressure']
                    X = {**condition['fuel'], **condition['oxidizer']}
                    
                    reduced_mech.TPX = T, P, X
                    # reduced_mech()               
                except Exception as e:
                    print(f"Mechanism validation failed: {str(e)}")
                    print(f"Failed condition: T={T}, P={P}, X={X}")
                    return float('inf'), None
                #reduced_mech()

                runner = SimulationRunner(reduced_mech, self.reactor_type)
    
                
                reduced_results = runner.run_simulation(condition) 
                print("Run simulation was called succesfully")
                
                ignition_delay = idt_value(
                    reduced_mech,
                    condition,
                    soln=False
                )
                
                print(f"Ignition delay time for reduced mech: {ignition_delay} ms")
                reduced_results["ignition_delay"] = ignition_delay
                try:
                    reduced_results["species_names"] = reduced_mech.species_names
                    species_reduced = reduced_mech.species_names
                    print(len(species_reduced), ": " , species_reduced)
                    
                    mole_fractions = {species: reduced_results["mole_fractions"][i] for i, species in enumerate(reduced_results["species_names"])}
                    reduced_results["mole_fractions"] = mole_fractions
                except Exception as e:
                    print(f"Error in evaluating reduced mechanism species: {e}")
                
                # Step 4: Run simulation with full mechanism
                if not hasattr(self, "full_results_cache"):
                    self.full_results_cache = {}
    
                if idx not in self.full_results_cache:
                    print(f"Running full mechanism simulation for condition {idx}...")
                    self.full_results_cache[idx] = self.full_mech.run_simulation(condition)
                    ignition_delay_full = idt_value(
                        self.mech,
                        condition,
                        soln=True
                    )
                    self.full_results_cache[idx]["ignition_delay"] = ignition_delay_full
                    print(f"Ignition delay time (IDT) for full mechanism (condition {idx}): {ignition_delay_full} ms")
                else:
                    print(f"Using cached results for full mechanism simulation (condition {idx})...")
                
                full_results = self.full_results_cache[idx]
                   
                fitness = self.fitness_calculator.combined_fitness(
                    reduced_results, full_results, 
                    self.key_species, sum(genome), self.genome_length)
                
                for key in total_fitness: 
                    total_fitness[key] += fitness[key]


            for key in total_fitness:
                total_fitness[key] /= num_conditions

            return total_fitness, reduced_results
        
        
        except Exception as e:
            print(f"Error during fitness evaluation for genome : {e}") 
            return {
            "combined_fitness": float("inf"),
            "temperature_fitness": float("inf"),
            "species_fitness": float("inf"),
            "ignition_delay_fitness": float("inf"),
            "reaction_count_fitness": float("inf")
        }, None

    def run_generation(self, population, generation, save_top_n):
        """
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
        fitness_data = []
        genome_runtimes = []
        species_usage = {}
        # Access the individuals array from the Population object
        individuals = population.individuals # gets array of population
        print(f"Processing {len(individuals)} individuals in generation {generation}")
      
            
        for i, genome in enumerate(individuals): #goes through all the genomes(individuals)
            start_time = time.time()
            fitness, reduced_results = self.evaluate_fitness(genome, generation)
            reaction_count = sum(genome)
            print("About to call fitness calcualtion")
            
                      
            fitness_data.append(fitness)    
           
            fitness_scores.append(fitness["combined_fitness"])
            print(f"Combined Fitness: {fitness["combined_fitness"]}")
            
            print(f"Genome Number: {len(fitness_scores)}")
            print(f"Fitness for generation {generation} genome number {len(fitness_scores)}: {fitness}")
            
            if reduced_results is not None:
                #count active reactions
                reaction_counts.append(reaction_count)
                print(f"Length of reaction_counts: {len(reaction_counts)}")
                
                max_temp = reduced_results.get("max_temperature", 0)
             
                result_entry= {
                    "fitness": fitness["combined_fitness"],
                    "reaction_count": reaction_count,
                    "max_temperature": max_temp,
                    "genome": genome.copy() if hasattr(genome, "copy") else list(genome),
                    "individual_index": i
                }
                all_results.append(result_entry)
        
        genome_runtime = time.time() - start_time
        genome_runtimes.append(genome_runtime)       
        # Sort results bby fitness
        all_results.sort(key=lambda x: x["fitness"])   
        
        # Calculate statistics
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else float('inf')
        min_fitness = min(fitness_scores) if fitness_scores else float('inf')
        max_fitness = max(fitness_scores) if fitness_scores else float('inf')
        std_fitness = np.std(fitness_scores) if len(fitness_scores) > 1 else 0

        avg_reactions = sum(reaction_counts) / len(reaction_counts) if reaction_counts else 0
        # min_reactions = min(reaction_counts) if reaction_counts else 0
        max_reactions = max(reaction_counts) if reaction_counts else 0      
        if fitness_scores:
            min_fitness_index = np.argmin(fitness_scores)  # Index of the genome with the minimum fitness
            min_fitness_genome = individuals[min_fitness_index]  # Genome with the minimum fitness
            min_reactions = sum(min_fitness_genome)  # Reaction count for the genome with the minimum fitness
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
                with open(f"{base_dir}/rank_{rank+1}_mechanism_info.json", 'w') as f:
                    json.dump({
                                "Fitness": result["fitness"],
                                "Reaction Count": result["reaction_count"],
                                "Individual Index": individual_idx
                            }, f, indent=4)

                    # Save active reaction indices
                    active_reactions = [i for i, active in enumerate(genome) if active]
                    json.dump({ "Active reaction indices": active_reactions
                                }, f, indent=4)
    
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
          
        return {
            "fitness_scores": fitness_scores,
            "fitness_data" : fitness_data,
            "genome_runtimes": genome_runtimes,
            "best_genome": all_results[0]["genome"] if all_results else None,
            "best_fitness": min_fitness,
            "active_reactions": min_reactions,
            "average_reactions": avg_reactions
    } 