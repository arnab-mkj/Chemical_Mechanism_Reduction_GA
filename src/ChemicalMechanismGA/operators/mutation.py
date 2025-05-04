import numpy as np

class Mutation:
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate
        
    def bit_flip_mutation(self, genome):
        #min_active_reactions = 50
        for i in range(len(genome)):
            if np.random.rand() < self.mutation_rate:
                genome[i] = 1 - genome[i]
        # if sum(genome) < min_active_reactions:
        #     genome[np.random.choice(len(genome))] = 1  # Reactivate a random reaction
        return genome
    
    def swap_pos_mutation(self, genome):
        mutated_genome = genome.copy()
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(len(genome), 2, replace = False)
            mutated_genome[idx1], mutated_genome[idx2] = mutated_genome[idx2], mutated_genome[idx1]
        return mutated_genome 