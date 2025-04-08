import numpy as np
import json
import time
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
                 fitness_evaluator, species_def):
        
        self.num_generations = num_generations
        self.fitness_history = []
        self.elite_size = elite_size
        self.fitness_evaluator = fitness_evaluator

        # Initialize population
        self.population = Population(population_size, genome_length, species_def)

        # Initialize operators
        self.selection = Selection()
        self.crossover = Crossover(crossover_rate)
        self.mutation = Mutation(mutation_rate)
        
        self.plotter = RealTimePlotter()

    def evolve(self, output_directory):
        """Main evolution loop."""
        
        overall_best_genome = None
        overall_best_fitness = float('inf')
        # self.plotter.show()  # Show the plot without blocking execution
        all_generation_stats = []
        last_improvement_generation = 0
        for generation in range(self.num_generations):
            start_time = time.time()  # Start timing the generation
            # Evaluate fitness for the current population
            self.current_generation = generation
            
            save_top_n = 3
            #returns a dictionary
            result = self.fitness_evaluator.run_generation(self.population, self.current_generation, save_top_n) # gets fitness scores for all genomes
            fitness_score = result["fitness_scores"]
            fitness_scores = np.array(fitness_score)  # Convert list to numpy array
            # Get statistics
            stats = self.population.get_statistics(fitness_scores)
            best_fitness = stats['best_fitness']
            self.fitness_history.append(best_fitness) # only stores the best fitness from a genration
            
            if generation > 0:
                fitness_improvement = self.fitness_history[-2] - best_fitness
            else:
                fitness_improvement = 0
                
            # Save the best genome as a YAML file
            best_genome, best_fitness = self.population.get_best_individual(fitness_scores)
            print(f"Overall Best genome of type: {type(best_genome)} for {generation}: {best_genome}")
            # Update the overall best if the current generation's best is better
            if best_fitness < overall_best_fitness:
                overall_best_fitness = best_fitness
                overall_best_genome = best_genome.copy()  # Make a copy to avoid reference issues
                last_improvement_generation = generation
                #print(f"New overall best fitness found in generation {generation + 1}: {overall_best_fitness}")
            
            #Calculate diversity metrics
            unique_genomes = len(set(tuple(self.population.get_individual(i)) for i in range(self.population.get_size())))
            genome_diversity = unique_genomes / self.population.get_size()

            # Calculate the minimum number of active reactions
            active_reactions = result["active_reactions"]  
            min_reactions = np.min(active_reactions)  # Minimum number of active reactions
            
            stats["min_reactions"] = min_reactions
            
            # Track crossover and mutation success rates
            total_crossovers = 0
            successful_crossovers = 0
            total_mutations = 0
            successful_mutations = 0

            # Create new population
            new_population = []
            population_size = self.population.get_size()
            

            # get the elite individuals
            elite_individuals = self.get_elite_individuals(fitness_scores)
            new_population.extend(elite_individuals)

            # Now fill the rest of the population through selection and variation
            #for _ in range(population_size // 2):
            while len(new_population) < population_size:
                # Select parents

                parent1_idx = self.selection.tournament_selection(fitness_scores)
                parent2_idx = self.selection.tournament_selection(fitness_scores)
                
                total_crossovers += 1
                # Apply crossover
                child1, child2 = self.crossover.single_point_crossover(
                    self.population.get_individual(parent1_idx),
                    self.population.get_individual(parent2_idx)
                )
                if not self.is_duplicate(child1, new_population):
                    successful_crossovers += 1
                    new_population.append(child1)
                if len(new_population) < population_size and not self.is_duplicate(child2, new_population):
                    successful_crossovers += 1
                    new_population.append(child2)

                # Apply mutation
                total_mutations += 2
                child1 = self.mutation.bit_flip_mutation(child1)
                child2 = self.mutation.bit_flip_mutation(child2)
                
                if not self.is_duplicate(child1, new_population):
                    successful_mutations += 1
                if len(new_population) < population_size and not self.is_duplicate(child2, new_population):
                    successful_mutations += 1

                # new_population.extend([child1, child2])

            # Replace old population
            self.population.replace_population(np.array(new_population[:population_size]))
            # Calculate success rates
            crossover_success_rate = successful_crossovers / total_crossovers if total_crossovers > 0 else 0
            mutation_success_rate = successful_mutations / total_mutations if total_mutations > 0 else 0

            # Calculate runtime for the generation
            generation_runtime = time.time() - start_time
            
            # Save generation stats
            generation_stats = {
                "generation": generation + 1,
                "best_fitness": stats['best_fitness'],
                "mean_fitness": stats['mean_fitness'],
                "min_reactions": min_reactions,
                "active_reactions_mean": stats['active_reactions_mean'],
                "fitness_improvement": fitness_improvement,
                "generations_since_improvement": generation - last_improvement_generation,
                "genome_diversity": genome_diversity,
                "crossover_success_rate": crossover_success_rate,
                "mutation_success_rate": mutation_success_rate,
                "generation_runtime": generation_runtime,
                "overall_best_genome": best_genome
            }
            all_generation_stats.append(generation_stats)
            
            # Save the stats to a JSON file after every generation
            with open(f"results/generation_stats.json", 'w') as f:
                json.dump(all_generation_stats, f, indent=2, default=self.convert_numpy_types)

            # Print generation stats
            print(f"Generation {generation + 1}")
            print(f"Best Fitness: {best_fitness}")
            print(f"Mean Fitness: {stats['mean_fitness']}")
            print(f"Min Reactions: {min_reactions}")
            print(f"Fitness Improvement: {fitness_improvement}")
            print(f"Genome Diversity: {genome_diversity:.2f}")
            print(f"Crossover Success Rate: {crossover_success_rate:.2f}")
            print(f"Mutation Success Rate: {mutation_success_rate:.2f}")
            print(f"Generation Runtime: {generation_runtime:.2f} seconds\n")

            self.plotter.update(generation, stats)     
        self.plotter.show()
        # return self.population.get_best_individual(fitness_score)
        return overall_best_genome, overall_best_fitness
    
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
    
    @staticmethod
    def convert_numpy_types(obj):
        """Helper function to convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)  # Convert numpy integers to Python int
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)  # Convert numpy floats to Python float
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        