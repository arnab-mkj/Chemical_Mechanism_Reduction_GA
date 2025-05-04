import numpy as np
import cantera as ct
import json

class Population:
    def __init__(self, popu_size, genome_length, species_def, deactivation_chance, init_with_reduced_mech):
        """
        Initialize Population class.

        Args:
            size (int): Size of the population
            genome_length (int): Length of each genome (number of reactions)
        """
        self.popu_size = popu_size
        self.genome_length = genome_length
        self.species_def = species_def
        self.deact  = deactivation_chance
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
                if np.random.rand() < self.deact:  # % chance of deactivating reactions
                    genome[i] = 1 - genome[i]
            population.append(genome)
        #print(f"The initial population: {population}")

        return np.array(population)



    def get_best_individual(self, fitness_scores):
        """
        Get the best individual from the population.

        Returns:
            tuple: (best_genome, best_fitness)
        """
        if fitness_scores is None:
            raise ValueError("Fitness scores have not been calculated yet")

        best_idx = np.argmin(fitness_scores) # returns index
        print("Type best_idx: ", best_idx)
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
        #print(f"new population: {list(self.individuals)}")
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


    def get_statistics(self, fitness_data, generation):
        # print(fitness_data)
        # Extract total fitness and components
        temperature_fitness_all = [d["temperature_fitness"] for d in fitness_data]
        species_fitness_all = [d["species_fitness"] for d in fitness_data]
        ignition_delay_fitness_all = [d["ignition_delay_fitness"] for d in fitness_data]
        reaction_fitness_all = [d["reaction_count_fitness"] for d in fitness_data]

        
        if not isinstance(fitness_data, list) or not all(isinstance(d, dict) for d in fitness_data):
            raise ValueError("Invalid fitness_data format. Expected a list of dictionaries.")
    
        combined_total = [d["combined_fitness"] for d in fitness_data]
        
        best_idx = np.argmin(combined_total)
        
        temperature_fitness = temperature_fitness_all[best_idx]
        species_fitness = species_fitness_all[best_idx]
        ignition_delay_fitness = ignition_delay_fitness_all[best_idx]
        reaction_fitness = reaction_fitness_all[best_idx]
    

        # Calculate statistics
        stats = {
            'generation': generation,
    
            
            'total_best_fitness': combined_total[best_idx],
            'temperature_fitness_min': temperature_fitness,
            'species_fitness_min': species_fitness,
            'ignition_delay_fitness_min': ignition_delay_fitness,
            'reaction_fitness_min': reaction_fitness,
            
            'worst_fitness': np.max(combined_total),
            'mean_fitness': np.mean(combined_total),
            'std_fitness': np.std(combined_total),
            
            'temperature_fitness_mean': np.mean(temperature_fitness_all),
            'species_fitness_mean': np.mean(species_fitness_all),
            'ignition_delay_fitness_mean': np.mean(ignition_delay_fitness_all),
            'reaction_fitness_mean': np.mean(reaction_fitness_all),
            
            'active_reactions_mean': np.mean([sum(genome) for genome in self.individuals]),
            'worst_reactions': np.max([sum(genome) for genome in self.individuals]),
            'min_reactions': np.min([sum(genome) for genome in self.individuals])
        }

        return stats