import numpy as np

class Population:
    def __init__(self, popu_size, genome_length):
        """
        Initialize Population class.

        Args:
            size (int): Size of the population
            genome_length (int): Length of each genome (number of reactions)
        """
        self.popu_size = popu_size
        self.genome_length = genome_length
        self.individuals = self.initialize_population()
        #self.fitness_scores = None

    def initialize_population(self):
        """
        Initialize population with all reactions active and slight diversity.

        Returns:
            np.array: Initial population
        """
        # Base genome with all reactions active
        base_genome = np.ones(self.genome_length, dtype=int)

        population = []
        for _ in range(self.popu_size):
            # Introduce slight diversity through random mutation
            genome = base_genome.copy()
            for i in range(self.genome_length):
                if np.random.rand() < 0.1:  # 10% chance of deactivating reactions
                    genome[i] = 1 - genome[i]
            population.append(genome)

        return np.array(population)



    def get_best_individual(self, fitness_scores):
        """
        Get the best individual from the population.

        Returns:
            tuple: (best_genome, best_fitness)
        """
        if fitness_scores is None:
            raise ValueError("Fitness scores have not been calculated yet")

        best_idx = np.argmin(fitness_scores)
        return self.individuals[best_idx], fitness_scores[best_idx]


    def replace_population(self, new_individuals):
        """
        Replace the current population with new individuals.

        Args:
            new_individuals (np.array): New population
        """
        if len(new_individuals) != len(self.individuals):
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
        return len(self.individuals)


    def get_statistics(self, fitness_scores):
        """
        Get population statistics.

        Returns:
            dict: Dictionary containing population statistics
        """
        if fitness_scores is None:
            raise ValueError("Fitness scores have not been calculated yet")

        return {
            'best_fitness': np.min(fitness_scores),
            'worst_fitness': np.max(fitness_scores),
            'mean_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'active_reactions_mean': np.mean([sum(genome) for genome in self.individuals])
        }