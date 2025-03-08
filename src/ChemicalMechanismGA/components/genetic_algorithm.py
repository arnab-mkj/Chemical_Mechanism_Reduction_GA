import numpy as np
import json
from src.ChemicalMechanismGA.operators.selection import Selection
from src.ChemicalMechanismGA.operators.crossover import Crossover
from src.ChemicalMechanismGA.operators.mutation import Mutation
from src.ChemicalMechanismGA.components.population import Population
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
from src.ChemicalMechanismGA.utils.visualization import RealTimePlotter    
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator

class GeneticAlgorithm:
    def __init__(self, population_size, genome_length, crossover_rate, 
                 mutation_rate, num_generations, elite_size, fitness_evaluator=None):
        
        self.num_generations = num_generations
        self.fitness_history = []
        self.elite_size = elite_size
        self.fitness_evaluator = fitness_evaluator

        # Initialize population
        self.population = Population(population_size, genome_length)

        # Initialize operators
        self.selection = Selection()
        self.crossover = Crossover(crossover_rate)
        self.mutation = Mutation(mutation_rate)
        
        self.plotter = RealTimePlotter()

    def evolve(self, output_directory):
        """Main evolution loop."""
        #best_results = None  # To store the best simulation results
        # if not isinstance(selected_plots, list):
        #     raise ValueError("selected_plots must be a list of plot options")
        overall_best_genome = None
        overall_best_fitness = float('inf')
        # self.plotter.show()  # Show the plot without blocking execution
        all_generation_stats = []
        for generation in range(self.num_generations):
            # Evaluate fitness for the current population
            self.current_generation = generation
            
            save_top_n = 3
            #returns a dictionary
            result = self.fitness_evaluator.run_generation(self.population, self.current_generation, save_top_n) # gets fitness scores for all genomes
            fitness_score = result["fitness_scores"]
            fitness_scores = np.array(fitness_score)  # Convert list to numpy array
            # Get statistics
            stats = self.population.get_statistics(fitness_scores)
            self.fitness_history.append(stats['best_fitness']) # only stores the best fitness from a genration

            print(f"Generation {generation + 1}")
            print(f"Best Fitness: {stats['best_fitness']}")
            print(f"Mean Fitness: {stats['mean_fitness']}")
            print(f"Active Reactions (mean): {stats['active_reactions_mean']:.2f}\n")
            
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()  # Convert numpy arrays to lists
                elif isinstance(obj, (np.int64, np.int32, np.int16)):
                    return int(obj)  # Convert numpy integers to Python int
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)  # Convert numpy floats to Python float
                else:
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
            
                    
            self.plotter.update(generation, stats)

            # Save the best genome as a YAML file
            best_genome, best_fitness = self.population.get_best_individual(fitness_scores)
            print(f"Overall Best genome of type: {type(best_genome)} for {generation}: {best_genome}")
            # Update the overall best if the current generation's best is better
            if best_fitness < overall_best_fitness:
                overall_best_fitness = best_fitness
                overall_best_genome = best_genome.copy()  # Make a copy to avoid reference issues
            
                #print(f"New overall best fitness found in generation {generation + 1}: {overall_best_fitness}")
            generation_stats = {
                "generation": generation + 1,
                "best_fitness": stats['best_fitness'],
                "mean_fitness": stats['mean_fitness'],
                "active_reactions_mean": stats['active_reactions_mean'],
                "overall_best_genome": best_genome
            }
            all_generation_stats.append(generation_stats)
            
            # Save the stats to a JSON file after every generation
            with open(f"results/generation_stats.json", 'w') as f:
                json.dump(all_generation_stats, f, indent=2, default=convert_numpy_types)
            
            #save_genome_as_yaml(best_genome, original_mechanism_path, output_path)

            # Create new population
            new_population = []
            population_size = self.population.get_size()
            
            print(type(fitness_scores))
            # get the elite individuals
            elite_individuals = self.get_elite_individuals(fitness_scores)
            new_population.extend(elite_individuals)

            # Now fill the rest of the population through selection and variation
            for _ in range(population_size // 2):
                # Select parents
                parent1_idx = self.selection.tournament_selection(fitness_scores)
                parent2_idx = self.selection.tournament_selection(fitness_scores)

                # Apply crossover
                child1, child2 = self.crossover.single_point_crossover(
                    self.population.get_individual(parent1_idx),
                    self.population.get_individual(parent2_idx)
                )

                # Apply mutation
                child1 = self.mutation.bit_flip_mutation(child1)
                child2 = self.mutation.bit_flip_mutation(child2)

                new_population.extend([child1, child2])

            # Replace old population
            self.population.replace_population(np.array(new_population[:population_size]))
        
        self.plotter.show()
        
        # return self.population.get_best_individual(fitness_score)
        return overall_best_genome, overall_best_fitness
    
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
        