import numpy as np
import json
import time
import matplotlib.pyplot as plt
from src.ChemicalMechanismGA.operators.selection import Selection
from src.ChemicalMechanismGA.operators.crossover import Crossover
from src.ChemicalMechanismGA.operators.mutation import Mutation
from src.ChemicalMechanismGA.components.population import Population
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
from src.ChemicalMechanismGA.utils.visualization import RealTimePlotter
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator

class GeneticAlgorithm:
    def __init__(self, population_size, genome_length, crossover_rate, 
                 mutation_rate, num_generations, elite_size, 
                 fitness_evaluator, species_def, difference_function, elitism_enabled, 
                 deactivation_chance, init_with_reduced_mech):
        
        self.num_generations = num_generations
        self.fitness_history = []
        self.elite_size = elite_size
        self.fitness_evaluator = fitness_evaluator
        self.crate = crossover_rate
        self.mrate = mutation_rate
        self.dfunc = difference_function
        self.e = elitism_enabled


        # Initialize population
        self.population = Population(population_size, genome_length, species_def, deactivation_chance, init_with_reduced_mech)

        # Initialize operators
        self.selection = Selection()
        self.crossover = Crossover(crossover_rate)
        self.mutation = Mutation(mutation_rate)
        
        self.plotter = RealTimePlotter()

    def evolve(self, output_directory):
        """Main evolution loop."""
        output_directory_location = f"{output_directory}/{self.dfunc}"
        overall_best_genome = None
        overall_best_fitness = float('inf')
        all_fitness_data =[]
        # self.plotter.show()  # Show the plot without blocking execution
        all_generation_stats = []
        generation_runtimes = []
        all_runtimes = []
        last_improvement_generation = 0
        
        for generation in range(self.num_generations):
            start_time = time.time()  # Start timing the generation
            # Evaluate fitness for the current population
            self.current_generation = generation
            
            save_top_n = 3
            #returns a dictionary
            result = self.fitness_evaluator.run_generation(self.population, self.current_generation, save_top_n) # gets fitness scores for all genomes
            all_fitness_data.extend(result["fitness_data"])
            fitness_scores = np.array(result["fitness_scores"])  # Convert list to numpy array
            
            for runtime in result["genome_runtimes"]:
                all_runtimes.append({"generation": generation + 1, "runtime": runtime})
            
            # Get statistics
            stats = self.population.get_statistics(all_fitness_data, generation)
            
            best_fitness = stats['total_best_fitness']
            self.fitness_history.append(best_fitness) # only stores the best fitness from a genration
            
            if generation > 0:
                fitness_improvement = self.fitness_history[-2] - best_fitness
            else:
                fitness_improvement = 0
                
            best_genome, best_fitness = self.population.get_best_individual(fitness_scores)
            best_fitness_reactions = sum(best_genome)
            print(f"Overall Best genome for {generation}: {best_genome}")
            # Update the overall best if the current generation's best is better
            if best_fitness < overall_best_fitness:
                overall_best_fitness = best_fitness
                overall_best_genome = best_genome.copy()
                current_best_reactions = best_fitness_reactions# Make a copy to avoid reference issues
                last_improvement_generation = generation
                #print(f"New overall best fitness found in generation {generation + 1}: {overall_best_fitness}")
            
            #Calculate diversity metrics
            unique_genomes = len(set(tuple(self.population.get_individual(i)) for i in range(self.population.get_size())))
            genome_diversity = unique_genomes / self.population.get_size()

            # Calculate the minimum number of active reactions
            active_reactions = result["active_reactions"]  
            min_reactions = np.min(active_reactions)  # Minimum number of active reactions
            stats["min_reactions"] = min_reactions
            
            stats["current_fitness_reactions"] = current_best_reactions
            # Track crossover and mutation success rates
            total_crossovers = 0
            successful_crossovers = 0
            total_mutations = 0
            successful_mutations = 0
            

            # Create new population
            new_population = []
            population_size = self.population.get_size()

            # get the elite individuals
            if self.e:
                elite_individuals = self.get_elite_individuals(fitness_scores)
                new_population.extend(elite_individuals)

            # Now fill the rest of the population through selection and variation
            #for _ in range(population_size // 2):
            while len(new_population) < population_size:
                # Select parents

                parent1_idx = self.selection.tournament_selection(fitness_scores)
                parent2_idx = self.selection.tournament_selection(fitness_scores)               
               
                # Apply crossover
                child1, child2 = self.crossover.single_point_crossover(
                    self.population.get_individual(parent1_idx),
                    self.population.get_individual(parent2_idx)
                )
                total_crossovers += 1
                if not self.is_duplicate(child1, new_population):
                    successful_crossovers += 1
                    new_population.append(child1)
                    
                if len(new_population) < population_size:
                    total_crossovers += 1
                    if not self.is_duplicate(child2, new_population):
                        successful_crossovers += 1
                        new_population.append(child2)

                # Apply mutation
                total_mutations += 1
                mutated_child1 = self.mutation.bit_flip_mutation(child1)
                if not self.is_duplicate(mutated_child1, new_population):
                    successful_mutations += 1
                    new_population.append(mutated_child1)

                if len(new_population) < population_size:
                    total_mutations += 1
                    mutated_child2 = self.mutation.bit_flip_mutation(child2)
                    if not self.is_duplicate(mutated_child2, new_population):
                        successful_mutations += 1
                        new_population.append(mutated_child2)

                # new_population.extend([child1, child2])

            # Replace old population
            self.population.replace_population(np.array(new_population[:population_size]))
            # Calculate success rates
            crossover_success_rate = successful_crossovers / total_crossovers if total_crossovers > 0 else 0
            mutation_success_rate = successful_mutations / total_mutations if total_mutations > 0 else 0

            # Calculate runtime for the generation
            generation_runtime = time.time() - start_time
            generation_runtimes.append(generation_runtime)
            # Save generation stats
            generation_stats = {
                "generation": generation,

                "best_fitness": stats['total_best_fitness'],
                "mean_fitness": stats['mean_fitness'],
                "worst_fitness": stats['worst_fitness'],
                "std_fitnes": stats['std_fitness'],
                "temperature_fitness_min" : stats['temperature_fitness_min'],
                "species_fitness_min" : stats['species_fitness_min'],
                "ignition_delay_fitness_min": stats['ignition_delay_fitness_min'],
                "reaction_fitness_min": stats['reaction_fitness_min'],
                "best_reactions": min_reactions,
                "active_reactions_mean": stats['active_reactions_mean'],
                "worst_reactions": stats['worst_reactions'],
                "min_reactions": stats['min_reactions'],
                "fitness_improvement": fitness_improvement,
                "generations_since_improvement": generation - last_improvement_generation,
                "genome_diversity": genome_diversity,
                "crossover_success_rate": crossover_success_rate,
                "mutation_success_rate": mutation_success_rate,
                "generation_runtime": generation_runtime,
                "overall_best_genome": best_genome,
                "overall_best_reactions":  stats["current_fitness_reactions"]
            }
            all_generation_stats.append(generation_stats)
            print("\n#################### All Generation Statistics ####################\n")

            print(f"Generation {generation_stats['generation']}")
            print(f"  Best Fitness: {generation_stats['best_fitness']:.6f}")
            print(f"  Mean Fitness: {generation_stats['mean_fitness']:.6f}")
            print(f"  Worst Fitness: {generation_stats['worst_fitness']:.6f}")
            print(f"  Fitness Standard Deviation: {generation_stats['std_fitnes']:.6f}")
            print(f"  Temperature Fitness Min: {generation_stats['temperature_fitness_min']:.6f}")
            print(f"  Species Fitness Min: {generation_stats['species_fitness_min']:.6f}")
            print(f"  Ignition Delay Fitness Min: {generation_stats['ignition_delay_fitness_min']:.6f}")
            print(f"  Reaction Fitness Min: {generation_stats['reaction_fitness_min']:.6f}")
            print(f"  Best Reactions: {generation_stats['best_reactions']}")
            print(f"  Active Reactions Mean: {generation_stats['active_reactions_mean']:.2f}")
            print(f"  Worst Reactions: {generation_stats['worst_reactions']}")
            print(f"  Minimum Reactions: {generation_stats['min_reactions']}")
            print(f"  Fitness Improvement: {generation_stats['fitness_improvement']:.6f}")
            print(f"  Generations Since Improvement: {generation_stats['generations_since_improvement']}")
            print(f"  Genome Diversity: {generation_stats['genome_diversity']:.2f}")
            print(f"  Crossover Success Rate: {generation_stats['crossover_success_rate']:.2f}")
            print(f"  Mutation Success Rate: {generation_stats['mutation_success_rate']:.2f}")
            print(f"  Generation Runtime: {generation_stats['generation_runtime']:.2f} seconds")
            print(f"  Overall Best Genome: {generation_stats['overall_best_genome']}")
            print(f"  Overall Best Reactions: {generation_stats['overall_best_reactions']}")
            print("------------------------------------------------------------\n")

            self.plotter.update(generation, stats)
            self.plotter.save(best_fitness, self.crate, self.mrate, self.dfunc)  
            
        self.plotter.show()
        self.print_and_save_generation_stats(all_generation_stats, output_directory_location)
        print("Runtime data saved successfully.")
    
        self.plot_runtime_scatter(all_runtimes)
        self.plot_generation_runtimes(generation_runtimes)
        # return self.population.get_best_individual(fitness_score)
        print(f"Best fitness of {overall_best_fitness} was found in generation {last_improvement_generation + 1}.")
        
        # self.plot_fitness_scatter(all_fitness_data)
        print("Scatter plots saved successfully.")
        
        return overall_best_genome, overall_best_fitness
    
    def print_and_save_generation_stats(self, all_generation_stats, output_directory):
        output_file=f"{output_directory}/generation_stats.json"
        serializable_data = [self.convert_numpy_types(gen_stats) for gen_stats in all_generation_stats]
        
        # Save all statistics to a JSON file
        with open(output_file, "w") as f:
            json.dump(serializable_data, f, indent=4)
        print(f"All generation statistics have been saved to {output_file}\n")
        
        
    def is_duplicate(self, genome, population):
        genome_tuple = tuple(genome)  # Convert genome to a hashable type
        for individual in population:
            if genome_tuple == tuple(individual):
                return True
        return False

    def get_elite_individuals(self, fitness_score):
        
        individuals = []
        for i in range(self.population.get_size()):
            individuals.append((
                self.population.get_individual(i),
                fitness_score[i]
            ))
        # Sort by fitness (lower is better)
        sorted_individuals = sorted(individuals, key=lambda x: x[1])
        
        # Return thr best individuals(just the genomes, not the fitness scores)
        return [ind[0] for ind in sorted_individuals[:self.elite_size]]
    
        
    def convert_numpy_types(self,obj):
        """
        Recursively convert numpy types to JSON-serializable types.
        Handles nested dictionaries and lists.
        """
        if isinstance(obj, dict):
            # Recursively process each key-value pair in the dictionary
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Recursively process each element in the list
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # Convert numpy arrays to lists
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16)):
            # Convert numpy integers to Python int
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            # Convert numpy floats to Python float
            return float(obj)
        else:
            # Return the object as-is if it's already JSON-serializable
            return obj
        
    def plot_generation_runtimes(self, generation_runtimes):
        """Plot the runtime for each generation."""
        generation_runtimes = [self.convert_numpy_types(runtime) for runtime in generation_runtimes]
        plt.figure(figsize=(10, 6))
        plt.scatter(range(1, len(generation_runtimes) + 1), generation_runtimes, marker='o', c='purple', cmap='viridis')
        plt.title("Generation-Wise Runtime", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Runtime (seconds)", fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.savefig(f"outputs/{self.dfunc}/generation_runtimes.png", bbox_inches="tight")
        plt.show()
        
    def plot_runtime_scatter(self, all_runtimes):
        """Plot a scatter plot of runtime vs. genome index for a generation."""
        all_runtimes = [self.convert_numpy_types(entry) for entry in all_runtimes]
        generations = [entry["generation"] for entry in all_runtimes]
        runtimes = [entry["runtime"] for entry in all_runtimes]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(generations, runtimes, cmap='viridis', c='blue', alpha=0.7)
        plt.title("Runtime vs Generation", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Runtime (seconds)", fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        # Save the plot
        plt.savefig(f"outputs/{self.dfunc}/genome_runtimes.png", bbox_inches="tight")
        plt.show()
        
    def plot_fitness_scatter(self, fitness_data):
        """Create scatter plots for fitness components vs total fitness."""
        fitness_data = [self.convert_numpy_types(entry) for entry in fitness_data]
        combined_fitness = [d["combined_fitness"] for d in fitness_data]
        temperature_fitness = [d["temperature_fitness"] for d in fitness_data]
        species_fitness = [d["species_fitness"] for d in fitness_data]
        ignition_delay_fitness = [d["ignition_delay_fitness"] for d in fitness_data]

        # Create scatter plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # Temperature Fitness vs Total Fitness
        axes[0].scatter(combined_fitness, temperature_fitness, alpha=0.7, color="blue")
        axes[0].set_title("Temperature Fitness vs Total Fitness")
        axes[0].set_xlabel("Total Fitness")
        axes[0].set_ylabel("Temperature Fitness")
        axes[0].grid(True)

        # Species Fitness vs Total Fitness
        axes[1].scatter(combined_fitness, species_fitness, alpha=0.7, color="green")
        axes[1].set_title("Species Fitness vs Total Fitness")
        axes[1].set_xlabel("Total Fitness")
        axes[1].set_ylabel("Species Fitness")
        axes[1].grid(True)

        # Ignition Delay Fitness vs Total Fitness
        axes[2].scatter(combined_fitness, ignition_delay_fitness, alpha=0.7, color="red")
        axes[2].set_title("Ignition Delay Fitness vs Total Fitness")
        axes[2].set_xlabel("Total Fitness")
        axes[2].set_ylabel("Ignition Delay Fitness")
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f"outputs/{self.dfunc}/fitness_scatter_plots.png")
        plt.show()