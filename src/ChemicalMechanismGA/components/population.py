import numpy as np
import cantera as ct
import json

class Population:
    def __init__(self, popu_size, genome_length, species_def, deactivation_chance, init_with_reduced_mech):
        """
        Initialize a population of chemical reaction mechanism genomes.
        
        Args:
            popu_size (int): Number of individuals in the population
            genome_length (int): Number of reactions (length of each genome)
            species_def (dict): Species definitions for mechanism initialization
            deactivation_chance (float): Probability [0-1] of randomly deactivating a reaction
            init_with_reduced_mech (bool): Whether to initialize with reduced mechanism
        """
        self.popu_size = popu_size
        self.genome_length = genome_length
        self.species_def = species_def
        self.deact = deactivation_chance  # Reaction deactivation probability
        self.individuals = self.initialize_population()  # Initialize population genomes

    def initialize_population(self):
        """
        Create initial population with controlled diversity.
        
        Returns:
            np.array: Array of genomes where:
                - Each genome is a binary array (1=active reaction, 0=inactive)
                - Initialized with all reactions active by default
                - Random deactivations applied based on deactivation_chance
        """
        # Base genome with all reactions initially active
        base_genome = np.ones(self.genome_length, dtype=int)

        population = []
        for _ in range(self.popu_size):
            genome = base_genome.copy()
            # Introduce controlled diversity through random deactivation
            for i in range(self.genome_length):
                if np.random.rand() < self.deact:  # Probabilistic deactivation
                    genome[i] = 1 - genome[i]  # Flip activation state
            population.append(genome)
            
        return np.array(population)

    def get_best_individual(self, fitness_scores):
        """
        Identify the best performing individual in the population.
        
        Args:
            fitness_scores (list/np.array): Fitness values for all individuals
            
        Returns:
            tuple: (best_genome, best_fitness) where:
                best_genome: Genome array of top performer
                best_fitness: Corresponding fitness score
                
        Raises:
            ValueError: If fitness scores haven't been calculated
        """
        if fitness_scores is None:
            raise ValueError("Fitness scores have not been calculated yet")

        best_idx = np.argmin(fitness_scores)  # Index of best fitness (minimization)
        return self.individuals[best_idx], fitness_scores[best_idx]

    def replace_population(self, new_individuals):
        """
        Replace current population with new generation.
        
        Args:
            new_individuals (np.array): New population genomes
            
        Raises:
            ValueError: If new population size doesn't match current size
        """
        if len(new_individuals) != len(self.individuals):
            raise ValueError(
                f"New population size ({len(new_individuals)}) "
                f"does not match required size ({self.popu_size})"
            )

        self.individuals = new_individuals

    def get_individual(self, index):
        """
        Retrieve specific individual from population.
        
        Args:
            index (int): Index of desired individual
            
        Returns:
            np.array: Genome of requested individual
        """
        return self.individuals[index]

    def get_size(self):
        """
        Get current population size.
        
        Returns:
            int: Number of individuals in population
        """
        return len(self.individuals)

    def get_statistics(self, fitness_data, generation):
        """
        Calculate comprehensive statistics about population performance.
        
        Args:
            fitness_data (list): List of fitness dictionaries for all evaluations
            generation (int): Current generation number
            
        Returns:
            dict: Comprehensive statistics including:
                - Fitness metrics (best, worst, mean, std)
                - Component fitness scores (temperature, species, etc.)
                - Reaction count statistics
                - Generation tracking
                
        Raises:
            ValueError: If fitness_data format is invalid
        """
        # Validate input format
        if not isinstance(fitness_data, list) or not all(isinstance(d, dict) for d in fitness_data):
            raise ValueError("Invalid fitness_data format. Expected a list of dictionaries.")
    
        # Extract all fitness components
        combined_total = [d["combined_fitness"] for d in fitness_data]
        temperature_fitness_all = [d["temperature_fitness"] for d in fitness_data]
        species_fitness_all = [d["species_fitness"] for d in fitness_data]
        ignition_delay_fitness_all = [d["ignition_delay_fitness"] for d in fitness_data]
        reaction_fitness_all = [d["reaction_count_fitness"] for d in fitness_data]
        
        # Identify best performing individual
        best_idx = np.argmin(combined_total)
        
        # Compile comprehensive statistics
        stats = {
            # Generation tracking
            'generation': generation,
            
            # Overall fitness metrics
            'total_best_fitness': combined_total[best_idx],
            'worst_fitness': np.max(combined_total),
            'mean_fitness': np.mean(combined_total),
            'std_fitness': np.std(combined_total),
            
            # Best component fitness scores
            'temperature_fitness_min': temperature_fitness_all[best_idx],
            'species_fitness_min': species_fitness_all[best_idx],
            'ignition_delay_fitness_min': ignition_delay_fitness_all[best_idx],
            'reaction_fitness_min': reaction_fitness_all[best_idx],
            
            # Average component fitness scores
            'temperature_fitness_mean': np.mean(temperature_fitness_all),
            'species_fitness_mean': np.mean(species_fitness_all),
            'ignition_delay_fitness_mean': np.mean(ignition_delay_fitness_all),
            'reaction_fitness_mean': np.mean(reaction_fitness_all),
            
            # Reaction count statistics
            'active_reactions_mean': np.mean([sum(genome) for genome in self.individuals]),
            'worst_reactions': np.max([sum(genome) for genome in self.individuals]),
            'min_reactions': np.min([sum(genome) for genome in self.individuals])
        }

        return stats