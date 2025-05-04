import numpy as np

class Crossover:
    def __init__(self, crossover_rate=None):
        """
        Initialize crossover operator with specified rate.
        
        Args:
            crossover_rate (float, optional): Probability of crossover occurring 
                                            between 0 and 1. Defaults to None.
        """
        self.crossover_rate = crossover_rate
        
    def single_point_crossover(self, parent1, parent2):
        """
        Perform single-point crossover between two parent genomes.
        
        Args:
            parent1 (np.array): First parent genome
            parent2 (np.array): Second parent genome
            
        Returns:
            tuple: Two child genomes (child1, child2)
        """
        # Only perform crossover with specified probability
        if np.random.rand() < self.crossover_rate:
            # Choose random crossover point
            point = np.random.choice(len(parent1), 1)[0]
            # Create children by swapping segments
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        # Return copies of parents if no crossover occurs
        return parent1.copy(), parent2.copy()
    
    def two_point_crossover(self, parent1, parent2):
        """
        Perform two-point crossover between two parent genomes.
        
        Args:
            parent1 (np.array): First parent genome
            parent2 (np.array): Second parent genome
            
        Returns:
            tuple: Two child genomes (child1, child2)
        """
        # Only perform crossover with specified probability
        if np.random.rand() < self.crossover_rate:
            # Choose two distinct crossover points
            points = sorted(np.random.choice(len(parent1), 2, replace=False))
            # Create children by swapping middle segment
            child1 = np.concatenate((parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]))
            child2 = np.concatenate((parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]))
            return child1, child2
        # Return copies of parents if no crossover occurs
        return parent1.copy(), parent2.copy()