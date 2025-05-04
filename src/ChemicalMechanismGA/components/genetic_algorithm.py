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
        """
        Initialize the genetic algorithm for chemical mechanism reduction.
        
        Args:
            population_size (int): Number of individuals in population
            genome_length (int): Length of genome representing reactions
            crossover_rate (float): Probability of crossover between parents
            mutation_rate (float): Probability of mutation per gene
            num_generations (int): Maximum number of generations to evolve
            elite_size (int): Number of top individuals preserved via elitism
            fitness_evaluator (FitnessEvaluator): Evaluates mechanism fitness
            species_def (dict): Species definitions for initialization
            difference_function (str): Method for fitness difference calculation
            elitism_enabled (bool): Whether to use elitism
            deactivation_chance (float): Probability of reaction deactivation
            init_with_reduced_mech (bool): Initialize with reduced mechanism
        """
        self.num_generations = num_generations
        self.fitness_history = []  # Tracks best fitness per generation
        self.elite_size = elite_size
        self.fitness_evaluator = fitness_evaluator
        self.crate = crossover_rate
        self.mrate = mutation_rate
        self.dfunc = difference_function
        self.e = elitism_enabled

        # Initialize population with given parameters
        self.population = Population(
            population_size, 
            genome_length, 
            species_def, 
            deactivation_chance, 
            init_with_reduced_mech
        )

        # Initialize genetic operators
        self.selection = Selection()  # Parent selection strategy
        self.crossover = Crossover(crossover_rate)  # Crossover operator
        self.mutation = Mutation(mutation_rate)  # Mutation operator
        
        # Visualization tool
        self.plotter = RealTimePlotter()

    def evolve(self, output_directory):
        """
        Execute the genetic algorithm evolution process.
        
        Args:
            output_directory (str): Path to save results
            
        Returns:
            tuple: (best_genome, best_fitness) found during evolution
        """
        output_directory_location = f"{output_directory}/{self.dfunc}"
        overall_best_genome = None
        overall_best_fitness = float('inf')  # Initialize with worst possible fitness
        all_fitness_data = []
        all_generation_stats = []
        generation_runtimes = []
        all_runtimes = []
        last_improvement_generation = 0  # Tracks stagnation
        
        # Main generational loop
        for generation in range(self.num_generations):
            start_time = time.time()
            self.current_generation = generation
            
            # Evaluate current population fitness
            save_top_n = 3  # Number of top performers to save
            result = self.fitness_evaluator.run_generation(
                self.population, 
                self.current_generation, 
                save_top_n
            )
            
            # Store fitness data and runtime metrics
            all_fitness_data.extend(result["fitness_data"])
            fitness_scores = np.array(result["fitness_scores"])
            
            # Track individual genome runtimes
            for runtime in result["genome_runtimes"]:
                all_runtimes.append({
                    "generation": generation + 1, 
                    "runtime": runtime
                })
            
            # Calculate population statistics
            stats = self.population.get_statistics(all_fitness_data, generation)
            best_fitness = stats['total_best_fitness']
            self.fitness_history.append(best_fitness)
            
            # Calculate fitness improvement
            if generation > 0:
                fitness_improvement = self.fitness_history[-2] - best_fitness
            else:
                fitness_improvement = 0
                
            # Get best individual of current generation
            best_genome, best_fitness = self.population.get_best_individual(fitness_scores)
            best_fitness_reactions = sum(best_genome)
            
            # Update overall best solution if improved
            if best_fitness < overall_best_fitness:
                overall_best_fitness = best_fitness
                overall_best_genome = best_genome.copy()
                current_best_reactions = best_fitness_reactions
                last_improvement_generation = generation
            
            # Calculate population diversity metrics
            unique_genomes = len(set(tuple(self.population.get_individual(i)) 
                for i in range(self.population.get_size())
            ))
            genome_diversity = unique_genomes / self.population.get_size()

            # Track reaction count statistics
            active_reactions = result["active_reactions"]  
            min_reactions = np.min(active_reactions)
            stats["min_reactions"] = min_reactions
            stats["current_fitness_reactions"] = current_best_reactions
            
            # Initialize operator success tracking
            total_crossovers = 0
            successful_crossovers = 0
            total_mutations = 0
            successful_mutations = 0

            # Create new population through selection and variation
            new_population = []
            population_size = self.population.get_size()

            # Apply elitism if enabled
            if self.e:
                elite_individuals = self.get_elite_individuals(fitness_scores)
                new_population.extend(elite_individuals)

            # Fill remaining population through selection, crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection for parents
                parent1_idx = self.selection.tournament_selection(fitness_scores)
                parent2_idx = self.selection.tournament_selection(fitness_scores)               
               
                # Single-point crossover
                child1, child2 = self.crossover.single_point_crossover(
                    self.population.get_individual(parent1_idx),
                    self.population.get_individual(parent2_idx)
                )
                
                # Track crossover success (non-duplicate offspring)
                total_crossovers += 1
                if not self.is_duplicate(child1, new_population):
                    successful_crossovers += 1
                    new_population.append(child1)
                    
                if len(new_population) < population_size:
                    total_crossovers += 1
                    if not self.is_duplicate(child2, new_population):
                        successful_crossovers += 1
                        new_population.append(child2)

                # Apply bit-flip mutation
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

            # Replace old population with new generation
            self.population.replace_population(np.array(new_population[:population_size]))
            
            # Calculate operator success rates
            crossover_success_rate = (successful_crossovers / total_crossovers 
                                     if total_crossovers > 0 else 0)
            mutation_success_rate = (successful_mutations / total_mutations 
                                   if total_mutations > 0 else 0)

            # Calculate and store generation runtime
            generation_runtime = time.time() - start_time
            generation_runtimes.append(generation_runtime)
            
            # Compile generation statistics
            generation_stats = {
                "generation": generation,
                "best_fitness": stats['total_best_fitness'],
                "mean_fitness": stats['mean_fitness'],
                "worst_fitness": stats['worst_fitness'],
                "std_fitnes": stats['std_fitness'],
                "temperature_fitness_min": stats['temperature_fitness_min'],
                "species_fitness_min": stats['species_fitness_min'],
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
                "overall_best_reactions": stats["current_fitness_reactions"]
            }
            all_generation_stats.append(generation_stats)
            
            # Print generation summary
            self.print_generation_stats(generation_stats)
            
            # Update visualization
            self.plotter.update(generation, stats)
            self.plotter.save(best_fitness, self.crate, self.mrate, self.dfunc)  

        # Finalization
        self.plotter.show()
        self.print_and_save_generation_stats(all_generation_stats, output_directory_location)
        self.plot_runtime_scatter(all_runtimes)
        self.plot_generation_runtimes(generation_runtimes)
        
        print(f"Best fitness of {overall_best_fitness} was found in generation {last_improvement_generation + 1}.")
        return overall_best_genome, overall_best_fitness

    def print_generation_stats(self, stats):
        """Print formatted generation statistics to console."""
        print("\n#################### Generation Statistics ####################\n")
        print(f"Generation {stats['generation']}")
        print(f"  Best Fitness: {stats['best_fitness']:.6f}")
        print(f"  Mean Fitness: {stats['mean_fitness']:.6f}")
        print(f"  Worst Fitness: {stats['worst_fitness']:.6f}")
        print(f"  Fitness Standard Deviation: {stats['std_fitnes']:.6f}")
        print(f"  Temperature Fitness Min: {stats['temperature_fitness_min']:.6f}")
        print(f"  Species Fitness Min: {stats['species_fitness_min']:.6f}")
        print(f"  Ignition Delay Fitness Min: {stats['ignition_delay_fitness_min']:.6f}")
        print(f"  Reaction Fitness Min: {stats['reaction_fitness_min']:.6f}")
        print(f"  Best Reactions: {stats['best_reactions']}")
        print(f"  Active Reactions Mean: {stats['active_reactions_mean']:.2f}")
        print(f"  Worst Reactions: {stats['worst_reactions']}")
        print(f"  Minimum Reactions: {stats['min_reactions']}")
        print(f"  Fitness Improvement: {stats['fitness_improvement']:.6f}")
        print(f"  Generations Since Improvement: {stats['generations_since_improvement']}")
        print(f"  Genome Diversity: {stats['genome_diversity']:.2f}")
        print(f"  Crossover Success Rate: {stats['crossover_success_rate']:.2f}")
        print(f"  Mutation Success Rate: {stats['mutation_success_rate']:.2f}")
        print(f"  Generation Runtime: {stats['generation_runtime']:.2f} seconds")
        print(f"  Overall Best Genome: {stats['overall_best_genome']}")
        print(f"  Overall Best Reactions: {stats['overall_best_reactions']}")
        print("------------------------------------------------------------\n")

    def print_and_save_generation_stats(self, all_generation_stats, output_directory):
        """Save generation statistics to JSON file."""
        output_file = f"{output_directory}/generation_stats.json"
        serializable_data = [
            self.convert_numpy_types(gen_stats) 
            for gen_stats in all_generation_stats
        ]
        
        with open(output_file, "w") as f:
            json.dump(serializable_data, f, indent=4)
        print(f"All generation statistics saved to {output_file}\n")
        
    def is_duplicate(self, genome, population):
        """Check if genome already exists in population."""
        genome_tuple = tuple(genome)
        for individual in population:
            if genome_tuple == tuple(individual):
                return True
        return False

    def get_elite_individuals(self, fitness_score):
        """Select top elite_size individuals based on fitness."""
        individuals = []
        for i in range(self.population.get_size()):
            individuals.append((
                self.population.get_individual(i),
                fitness_score[i]
            ))
        # Sort by fitness (ascending - lower is better)
        sorted_individuals = sorted(individuals, key=lambda x: x[1])
        return [ind[0] for ind in sorted_individuals[:self.elite_size]]
    
    def convert_numpy_types(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object potentially containing numpy types
            
        Returns:
            Object with numpy types converted to native Python types
        """
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
        
    def plot_generation_runtimes(self, generation_runtimes):
        """Plot runtime per generation."""
        generation_runtimes = [
            self.convert_numpy_types(runtime) 
            for runtime in generation_runtimes
        ]
        plt.figure(figsize=(10, 6))
        plt.scatter(
            range(1, len(generation_runtimes) + 1), 
            generation_runtimes, 
            marker='o', 
            c='purple'
        )
        plt.title("Generation-Wise Runtime", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Runtime (seconds)", fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.savefig(f"outputs/{self.dfunc}/generation_runtimes.png", bbox_inches="tight")
        plt.show()
        
    def plot_runtime_scatter(self, all_runtimes):
        """Plot runtime vs generation for all genomes."""
        all_runtimes = [
            self.convert_numpy_types(entry) 
            for entry in all_runtimes
        ]
        generations = [entry["generation"] for entry in all_runtimes]
        runtimes = [entry["runtime"] for entry in all_runtimes]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(generations, runtimes, c='blue', alpha=0.7)
        plt.title("Runtime vs Generation", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Runtime (seconds)", fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.savefig(f"outputs/{self.dfunc}/genome_runtimes.png", bbox_inches="tight")
        plt.show()