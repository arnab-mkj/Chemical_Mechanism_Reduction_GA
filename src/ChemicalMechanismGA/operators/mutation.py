import numpy as np

class Mutation:
    def __init__(self, mutation_rate):
        """
        Initialize mutation operator with specified rate.
        
        Args:
            mutation_rate (float): Probability of mutation per gene between 0 and 1
        """
        self.mutation_rate = mutation_rate
        
    def bit_flip_mutation(self, genome):
        """
        Perform bit-flip mutation on a genome.
        
        Args:
            genome (np.array): Genome to mutate
            
        Returns:
            np.array: Mutated genome
        """
        # Iterate through each gene
        for i in range(len(genome)):
            # Flip bit with mutation probability
            if np.random.rand() < self.mutation_rate:
                genome[i] = 1 - genome[i]
        return genome
    
    def swap_pos_mutation(self, genome):
        """
        Perform position swap mutation on a genome.
        
        Args:
            genome (np.array): Genome to mutate
            
        Returns:
            np.array: Mutated genome
        """
        mutated_genome = genome.copy()
        # Perform swap with mutation probability
        if np.random.rand() < self.mutation_rate:
            # Select two distinct positions
            idx1, idx2 = np.random.choice(len(genome), 2, replace=False)
            # Swap their values
            mutated_genome[idx1], mutated_genome[idx2] = mutated_genome[idx2], mutated_genome[idx1]
        return mutated_genome