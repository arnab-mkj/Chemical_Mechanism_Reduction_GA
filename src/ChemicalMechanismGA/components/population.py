import numpy as np

class Population:
    def __init__(self, size, genome_length):
        """
        Initialize Population class.

        Args:
            size (int): Size of the population
            genome_length (int): Length of each genome (number of reactions)
        """
        self.size = size
        self.genome_length = genome_length
        self.individuals = self.initialize_population()
        self.fitness_scores = None

    def initialize_population(self):
        """
        Initialize population with all reactions active and slight diversity.

        Returns:
            np.array: Initial population
        """
        # Base genome with all reactions active
        base_genome = np.ones(self.genome_length, dtype=int)

        population = []
        for _ in range(self.size):
            # Introduce slight diversity through random mutation
            genome = base_genome.copy()
            for i in range(self.genome_length):
                if np.random.rand() < 0.1:  # 10% chance of deactivating reactions
                    genome[i] = 1 - genome[i]
            population.append(genome)

        return np.array(population)


    def evaluate_population_fitness(self, fitness_function):
        """
        Evaluate fitness for all individuals in the population.

        Args:
            fitness_function (callable): Function to evaluate fitness
        """
        fitness_values = [fitness_function(genome) for genome in self.individuals]
        self.fitness_scores = np.array(fitness_values) # stores the fitness values


    def get_best_individual(self):
        """
        Get the best individual from the population.

        Returns:
            tuple: (best_genome, best_fitness)
        """
        if self.fitness_scores is None:
            raise ValueError("Fitness scores have not been calculated yet")

        best_idx = np.argmin(self.fitness_scores)
        return self.individuals[best_idx], self.fitness_scores[best_idx]


    def replace_population(self, new_individuals):
        """
        Replace the current population with new individuals.

        Args:
            new_individuals (np.array): New population
        """
        if len(new_individuals) != self.size:
            raise ValueError(f"New population size ({len(new_individuals)}) does not match required size ({self.size})")

        self.individuals = new_individuals
        self.fitness_scores = None  # Reset fitness scores


    def get_individual(self, index):
        """
        Get individual at specified index.

        Args:
            index (int): Index of individual

        Returns:
            np.array: Individual genome
        """
        return self.individuals[index]


    def get_size(self):
        """
        Get population size.

        Returns:
            int: Population size
        """
        return self.size


    def get_statistics(self):
        """
        Get population statistics.

        Returns:
            dict: Dictionary containing population statistics
        """
        if self.fitness_scores is None:
            raise ValueError("Fitness scores have not been calculated yet")

        return {
            'best_fitness': np.min(self.fitness_scores),
            'worst_fitness': np.max(self.fitness_scores),
            'mean_fitness': np.mean(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores),
            'active_reactions_mean': np.mean([sum(genome) for genome in self.individuals])
        }