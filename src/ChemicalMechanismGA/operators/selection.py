import numpy as np

class Selection:
    
    @staticmethod
    def tournament_selection(fitness_scores):
        
        indices = np.random.choice(len(fitness_scores), 5, replace = False)
        return indices[np.argmin(fitness_scores[indices])] # returns the indices of the selected positions
    
    
    
    @staticmethod
    def roulette_wheel_selection(fitness_scores):
 
        total_fit = sum(fitness_scores)
        prob = [f/total_fit for f in fitness_scores]
        return np.random.choice(len(fitness_scores), p =prob)
        