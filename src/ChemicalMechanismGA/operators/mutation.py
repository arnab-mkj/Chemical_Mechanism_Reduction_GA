import numpy as np

class Mutation:
    def __init__(self, mutation_rate=0.1):
        self.mutation_rate = mutation_rate
        
    def bit_flip_mutation(self, genome):
        mutated_genome = genome.copy()
        for i in range(len(mutated_genome)):
            if np.random.rand() < self.mutation_rate:
                mutated_genome[i] = 1 - mutated_genome[i]
        return mutated_genome
    
    def swap_pos_mutation(self, genome):
        mutated_genome = genome.copy()
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(len(genome), 2, replace = False)
            mutated_genome[idx1], mutated_genome[idx2] = mutated_genome[idx2], mutated_genome[idx1]
        return mutated_genome 