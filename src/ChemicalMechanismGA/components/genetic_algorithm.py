import numpy as np
from ..operators.selection import Selection
from ..operators.crossover import Crossover
from ..operators.mutation import Mutation
from .population import Population
from ..utils.save_best_genome import save_genome_as_yaml

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

    def evolve(self, fitness_function, original_mechanism_path, output_directory, reactor_type):
        """Main evolution loop."""
        for generation in range(self.num_generations):
            # Evaluate fitness for the current population
            self.population.evaluate_fitness(
                lambda genome: fitness_function(
                    genome=genome,
                    original_mechanism_path=original_mechanism_path,
                    reactor_type=reactor_type
                )
            )

            # Get statistics
            stats = self.population.get_statistics()
            self.fitness_history.append(stats['best_fitness'])

            print(f"Generation {generation + 1}")
            print(f"Best Fitness: {stats['best_fitness']}")
            print(f"Mean Fitness: {stats['mean_fitness']}")
            print(f"Active Reactions (mean): {stats['active_reactions_mean']:.2f}\n")

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
        self.population.evaluate_fitness(
            lambda genome: fitness_function(
                genome=genome,
                original_mechanism_path=original_mechanism_path,
                reactor_type=reactor_type
            )
        )
        
        return self.population.get_best_individual()