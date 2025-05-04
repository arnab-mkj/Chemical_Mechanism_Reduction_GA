import cantera as ct
import numpy as np
import json
import matplotlib.pyplot as plt
import time
import os
from scipy import integrate
from src.ChemicalMechanismGA.components.simulation_runner import SimulationRunner
from ..utils.save_species_conc import save_mole_fractions_to_json, save_species_concentrations
from src.ChemicalMechanismGA.components.error_fitness import ConstantPressureFitness
from src.ChemicalMechanismGA.components.IDT_calc import idt_value
import math

class FitnessEvaluator:
    def __init__(self, mech, reactor_type, conditions,
                 weights, genome_length, difference_function, sharpening_factor, normalization_method, key_species):
        """
        Initializes the fitness evaluator for chemical mechanism reduction.
        
        Args:
            mech (str): Path to chemical mechanism file
            reactor_type (str): Type of reactor ('constant_pressure', etc.)
            conditions (list): List of condition dictionaries for evaluation
            weights (dict): Weighting factors for fitness components
            genome_length (int): Length of genome representing full mechanism
            difference_function (str): Method for difference calculation
            sharpening_factor (float): Parameter for fitness normalization
            normalization_method (str): Normalization method ('sigmoid', etc.)
            key_species (list): List of important species to track
        """
        self.mech = mech
        self.reactor_type = reactor_type
        self.conditions = conditions
        self.genome_length = genome_length
        self.dfunc = difference_function
        self.norm_func = normalization_method
        self.sharpening_factor = sharpening_factor
        self.weights = weights
        self.key_species = key_species
        
        # Initialize simulation runner for full mechanism
        self.full_mech = SimulationRunner(self.mech, self.reactor_type)
        
        # Set up appropriate fitness calculator
        if reactor_type == "constant_pressure":
            self.fitness_calculator = ConstantPressureFitness(
                self.dfunc, 
                self.sharpening_factor, 
                self.norm_func, 
                self.conditions, 
                self.weights
            )
        else:
            raise ValueError(f"Unsupported reactor type: {reactor_type}")

    def create_reduced_mechanism(self, genome, write_to_file, keep_all_species, condition):
        """
        Creates a reduced mechanism based on genome activation patterns.
        
        Args:
            genome (list): Binary list indicating active reactions
            write_to_file (bool): Whether to save reduced mechanism
            keep_all_species (bool): Whether to keep unused species
            condition (dict): Current simulation condition
            
        Returns:
            tuple: (reduced_mech, file_path) where:
                reduced_mech: Cantera Solution object
                file_path: Path to saved mechanism (empty if not saved)
        """
        gas = ct.Solution(self.mech)
        reactions = gas.reactions()
        
        # Handle duplicate reactions (must be kept/removed together)
        duplicate_groups = {}
        for i, reaction in enumerate(reactions):
            if hasattr(reaction, 'duplicate') and reaction.duplicate:
                key = tuple(sorted(reaction.reactants.items()))
                if key not in duplicate_groups:
                    duplicate_groups[key] = []
                duplicate_groups[key].append(i)
                
        # Ensure consistent treatment of duplicate reactions        
        for group in duplicate_groups.values():
            if any(genome[i] == 1 for i in group):
                for i in group:
                    genome[i] = 1  # Include all duplicates
            else:
                for i in group:
                    genome[i] = 0  # Exclude all duplicates
            
        # Create reduced reaction list
        reduced_reactions = [reaction for i, reaction in enumerate(reactions) if genome[i] == 1]
        if len(reduced_reactions) < 50:
            raise ValueError("Reduced mechanism has too few reactions")
        
        # Handle species selection
        if keep_all_species:
            reduced_species = gas.species()
        else:
            # Track species actually used in reactions
            species_used = set()
            for reaction in reduced_reactions:
                species_used.update(reaction.reactants.keys())
                species_used.update(reaction.products.keys())
                
            # Always keep fuel and oxidizer species    
            fuel_species = condition['fuel'].keys()
            oxidizer_species = condition['oxidizer'].keys()
            species_used.update(fuel_species)
            species_used.update(oxidizer_species)
            
            reduced_species = [sp for sp in gas.species() if sp.name in species_used]
        
        # Create reduced mechanism solution
        reduced_mech = ct.Solution(
            thermo="IdealGas",
            kinetics="GasKinetics",
            transport="mixture-averaged",
            species=reduced_species,
            reactions=reduced_reactions,
        )  
        
        # Validate all reaction species exist
        for reaction in reduced_reactions:
            reaction_species = set(reaction.reactants.keys()).union(reaction.products.keys())
            for sp in reaction_species:
                if sp not in reduced_mech.species_names:
                    raise ValueError(f"Reaction {reaction.equation} references missing species {sp}")
       
        # Optional file output
        file_path = ""
        if write_to_file:
            file_path = f"reduced_mech_{len(reduced_reactions)}_rxns.yaml"
            reduced_mech.write_yaml(file_path)
            print(f"Reduced mechanism written to {file_path}")
        
        print(f"Created reduced mechanism with {len(reduced_reactions)} reactions and {len(reduced_mech.species_names)} species.")
        return reduced_mech, file_path
    
    def evaluate_fitness(self, genome, generation):
        """
        Evaluates fitness of a genome across all conditions.
        
        Args:
            genome (list): Binary genome representing active reactions
            generation (int): Current generation number
            
        Returns:
            tuple: (total_fitness, reduced_results) where:
                total_fitness: Dictionary of fitness metrics
                reduced_results: Simulation results from reduced mechanism
        """
        try:
            total_fitness = {
                "combined_fitness": 0.0,
                "temperature_fitness": 0.0,
                "species_fitness": 0.0,
                "ignition_delay_fitness": 0.0,
                "reaction_count_fitness": 0.0
            }
            num_conditions = len(self.conditions)
            reduced_results = {}
            
            for idx, condition in enumerate(self.conditions):
                try:
                    # Create reduced mechanism for current condition
                    reduced_mech, _ = self.create_reduced_mechanism(
                        genome, 
                        write_to_file=False,
                        keep_all_species=False, 
                        condition=condition
                    )
                    
                    # Set initial state
                    T = condition['temperature']
                    P = condition['pressure']
                    X = {**condition['fuel'], **condition['oxidizer']}
                    reduced_mech.TPX = T, P, X
                    
                except Exception as e:
                    print(f"Mechanism validation failed: {str(e)}")
                    print(f"Failed condition: T={T}, P={P}, X={X}")
                    return float('inf'), None
                
                # Run simulation with reduced mechanism
                runner = SimulationRunner(reduced_mech, self.reactor_type)
                reduced_results = runner.run_simulation(condition) 
                
                # Calculate ignition delay
                ignition_delay = idt_value(reduced_mech, condition, soln=False)
                reduced_results["ignition_delay"] = ignition_delay
                
                # Process species data
                try:
                    reduced_results["species_names"] = reduced_mech.species_names
                    mole_fractions = {
                        species: reduced_results["mole_fractions"][i] 
                        for i, species in enumerate(reduced_results["species_names"])
                    }
                    reduced_results["mole_fractions"] = mole_fractions
                except Exception as e:
                    print(f"Error processing species data: {e}")
                
                # Run/cache full mechanism results
                if not hasattr(self, "full_results_cache"):
                    self.full_results_cache = {}
    
                if idx not in self.full_results_cache:
                    self.full_results_cache[idx] = self.full_mech.run_simulation(condition)
                    ignition_delay_full = idt_value(self.mech, condition, soln=True)
                    self.full_results_cache[idx]["ignition_delay"] = ignition_delay_full
                
                full_results = self.full_results_cache[idx]
                
                # Calculate fitness components
                fitness = self.fitness_calculator.combined_fitness(
                    reduced_results, 
                    full_results, 
                    self.key_species, 
                    sum(genome), 
                    self.genome_length
                )
                
                # Accumulate fitness across conditions
                for key in total_fitness: 
                    total_fitness[key] += fitness[key]

            # Average fitness across conditions
            for key in total_fitness:
                total_fitness[key] /= num_conditions

            return total_fitness, reduced_results
        
        except Exception as e:
            print(f"Error during fitness evaluation: {e}") 
            return {
                "combined_fitness": float("inf"),
                "temperature_fitness": float("inf"),
                "species_fitness": float("inf"),
                "ignition_delay_fitness": float("inf"),
                "reaction_count_fitness": float("inf")
            }, None

    def run_generation(self, population, generation, save_top_n):
        """
        Evaluates an entire generation of genomes.
        
        Args:
            population (Population): Population object containing genomes
            generation (int): Current generation number
            save_top_n (int): Number of top performers to save
            
        Returns:
            dict: Results dictionary containing:
                fitness_scores: List of combined fitness scores
                fitness_data: Detailed fitness metrics
                genome_runtimes: Execution times per genome
                best_genome: Genome with best fitness
                best_fitness: Best fitness value
                active_reactions: Reaction count of best genome
                average_reactions: Average reaction count
        """
        # Setup output directory
        base_dir = f"results/generation_{generation}"
        os.makedirs(base_dir, exist_ok=True)

        # Initialize tracking variables
        fitness_scores = []
        all_results = []
        reaction_counts = []
        fitness_data = []
        genome_runtimes = []
        
        # Process each genome in population
        individuals = population.individuals
        print(f"Processing {len(individuals)} individuals in generation {generation}")
      
        for i, genome in enumerate(individuals):
            start_time = time.time()
            
            # Evaluate fitness
            fitness, reduced_results = self.evaluate_fitness(genome, generation)
            reaction_count = sum(genome)
            
            # Store results
            fitness_data.append(fitness)    
            fitness_scores.append(fitness["combined_fitness"])
            reaction_counts.append(reaction_count)
            
            # Track detailed results for top performers
            if reduced_results is not None:
                max_temp = reduced_results.get("max_temperature", 0)
                result_entry = {
                    "fitness": fitness["combined_fitness"],
                    "reaction_count": reaction_count,
                    "max_temperature": max_temp,
                    "genome": genome.copy() if hasattr(genome, "copy") else list(genome),
                    "individual_index": i
                }
                all_results.append(result_entry)
            
            genome_runtimes.append(time.time() - start_time)
        
        # Calculate statistics
        all_results.sort(key=lambda x: x["fitness"])
        avg_fitness = np.mean(fitness_scores) if fitness_scores else float('inf')
        min_fitness = min(fitness_scores) if fitness_scores else float('inf')
        max_fitness = max(fitness_scores) if fitness_scores else float('inf')
        std_fitness = np.std(fitness_scores) if len(fitness_scores) > 1 else 0
        avg_reactions = np.mean(reaction_counts) if reaction_counts else 0
        
        # Get reaction count of best genome
        min_reactions = sum(individuals[np.argmin(fitness_scores)]) if fitness_scores else 0
        
        # Save top performers' data
        top_n_results = all_results[:save_top_n]
        for rank, result in enumerate(top_n_results):
            individual_idx = result["individual_index"]
            
            # Save species concentrations
            if "mole_fractions" in result:
                species_names = result.get("species_names", [])
                filename = f"{base_dir}/rank_{rank+1}_individual_{individual_idx}.json"
                save_mole_fractions_to_json(result, species_names, generation, filename)
                
            # Save mechanism info
            with open(f"{base_dir}/rank_{rank+1}_mechanism_info.json", 'w') as f:
                json.dump({
                    "Fitness": result["fitness"],
                    "Reaction Count": result["reaction_count"],
                    "Individual Index": individual_idx,
                    "Active reaction indices": [i for i, active in enumerate(result["genome"]) if active]
                }, f, indent=4)
    
        return {
            "fitness_scores": fitness_scores,
            "fitness_data": fitness_data,
            "genome_runtimes": genome_runtimes,
            "best_genome": all_results[0]["genome"] if all_results else None,
            "best_fitness": min_fitness,
            "active_reactions": min_reactions,
            "average_reactions": avg_reactions
        }