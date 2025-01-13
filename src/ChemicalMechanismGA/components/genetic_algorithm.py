import numpy as np
from src.ChemicalMechanismGA.operators.selection import Selection
from src.ChemicalMechanismGA.operators.crossover import Crossover
from src.ChemicalMechanismGA.operators.mutation import Mutation
from src.ChemicalMechanismGA.components.population import Population
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
from src.ChemicalMechanismGA.utils.visualization import RealTimePlotter    

class GeneticAlgorithm:
    def __init__(self, population_size, genome_length, crossover_rate=0.8, 
                 mutation_rate=0.1, num_generations=100):
        self.num_generations = num_generations
        self.fitness_history = []

        # Initialize population
        self.population = Population(population_size, genome_length)

        # Initialize operators
        self.selection = Selection()
        self.crossover = Crossover(crossover_rate)
        self.mutation = Mutation(mutation_rate)
        
        self.plotter = RealTimePlotter()

    def evolve(self, fitness_function, original_mechanism_path, output_directory, reactor_type):
        """Main evolution loop."""
        #best_results = None  # To store the best simulation results
        
        for generation in range(self.num_generations):
            # Evaluate fitness for the current population
            self.population.evaluate_population_fitness(
                lambda genome: fitness_function(
                    genome=genome,
                    original_mechanism_path=original_mechanism_path,
                    reactor_type=reactor_type,
                    generation=generation,
                    filename=None
                )
            )

            # Get statistics
            stats = self.population.get_statistics()
            self.fitness_history.append(stats['best_fitness'])

            print(f"Generation {generation + 1}")
            print(f"Best Fitness: {stats['best_fitness']}")
            print(f"Mean Fitness: {stats['mean_fitness']}")
            print(f"Active Reactions (mean): {stats['active_reactions_mean']:.2f}\n")
            
            self.plotter.update(generation, stats)

            # Save the best genome as a YAML file
            best_genome, best_fitness = self.population.get_best_individual()
            
            output_path = f"{output_directory}/reduced_mechanism_gen{generation+1}.yaml"
            #save_genome_as_yaml(best_genome, original_mechanism_path, output_path)

            # Create new population
            new_population = []
            population_size = self.population.get_size()

            for _ in range(population_size // 2):
                # Select parents
                parent1_idx = self.selection.tournament_selection(self.population.fitness_scores)
                parent2_idx = self.selection.tournament_selection(self.population.fitness_scores)

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

        # Final evaluation
        print("Starting final evaluation of the population...")
        self.population.evaluate_population_fitness(
            lambda genome: fitness_function(
                genome=genome,
                original_mechanism_path=original_mechanism_path,
                reactor_type=reactor_type
            ) #extracts only the fitness funtions
        )
        print("Final evaluation completed.")
        
        self.plotter.show()
        
        return self.population.get_best_individual()