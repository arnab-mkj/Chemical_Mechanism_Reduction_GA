import numpy as np

class Crossover:
    def __init__(self, crossover_rate = None):
        self.crossover_rate = crossover_rate
        
    def single_point_crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.choice(len(parent1), 1)[0]
            child1 = np.concatenate((parent1[:point] , parent2[point:]))
            child2 = np.concatenate((parent2[:point] , parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def two_point_crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            points = sorted(np.random.choice(len(parent1), 2, replace = False))
            child1 = np.concatenate((parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:] ))
            child2 = np.concatenate((parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:] ))
            return child1, child2
        return parent1.copy(), parent2.copy()