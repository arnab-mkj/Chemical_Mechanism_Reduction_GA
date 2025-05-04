import numpy as np

class Selection:
    @staticmethod
    def tournament_selection(fitness_scores):
        """
        Perform tournament selection from a population.
        
        Args:
            fitness_scores (np.array): Array of fitness scores for population
            
        Returns:
            int: Index of selected individual
        """
        # Randomly select tournament participants
        indices = np.random.choice(len(fitness_scores), 5, replace=False)
        # Return index of best performer (min fitness)
        return indices[np.argmin(fitness_scores[indices])]
    
    @staticmethod
    def roulette_wheel_selection(fitness_scores):
        """
        Perform roulette wheel selection from a population.
        
        Args:
            fitness_scores (np.array): Array of fitness scores for population
            
        Returns:
            int: Index of selected individual
        """
        # Convert fitness scores to selection probabilities
        total_fit = sum(fitness_scores)
        prob = [f/total_fit for f in fitness_scores]
        # Select based on probability distribution
        return np.random.choice(len(fitness_scores), p=prob)