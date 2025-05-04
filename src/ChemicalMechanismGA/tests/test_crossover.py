import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Crossover class directly in the file
class Crossover:
    def __init__(self, crossover_rate=None):
        """
        Initialize crossover operator with specified rate.
        
        Args:
            crossover_rate (float, optional): Probability of crossover occurring 
                                            between 0 and 1. Defaults to None.
        """
        self.crossover_rate = crossover_rate
        
    def single_point_crossover(self, parent1, parent2, point=None):
        """
        Perform single-point crossover between two parent genomes.
        
        Args:
            parent1 (np.array): First parent genome
            parent2 (np.array): Second parent genome
            point (int, optional): Specific crossover point (if None, chosen randomly)
            
        Returns:
            tuple: Two child genomes (child1, child2), crossover point
        """
        # Only perform crossover with specified probability
        if np.random.rand() < self.crossover_rate:
            # Choose random crossover point if not specified
            if point is None:
                point = np.random.choice(len(parent1), 1)[0]
            # Create children by swapping segments
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2, point
        # Return copies of parents if no crossover occurs
        return parent1.copy(), parent2.copy(), None
    
    def two_point_crossover(self, parent1, parent2, points=None):
        """
        Perform two-point crossover between two parent genomes.
        
        Args:
            parent1 (np.array): First parent genome
            parent2 (np.array): Second parent genome
            points (list, optional): Specific crossover points (if None, chosen randomly)
            
        Returns:
            tuple: Two child genomes (child1, child2), crossover points
        """
        # Only perform crossover with specified probability
        if np.random.rand() < self.crossover_rate:
            # Choose two distinct crossover points if not specified
            if points is None:
                points = sorted(np.random.choice(len(parent1), 2, replace=False))
            # Create children by swapping middle segment
            child1 = np.concatenate((parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]))
            child2 = np.concatenate((parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]))
            return child1, child2, points
        # Return copies of parents if no crossover occurs
        return parent1.copy(), parent2.copy(), None

# Test case to visualize crossover behavior
def run_test(crossover_rate, genome_length, population_size):
    # Test parameters
    
    num_trials = 500  # Number of crossover trials per rate
    crossover_rates = [crossover_rate]  # Different crossover rates to test
    
    # Generate a population of random binary parent genomes
    population = np.random.randint(0, 2, size=(population_size, genome_length))
    
    # Initialize arrays to track crossover frequencies for each position
    # Shape: (number of rates, genome length)
    single_point_freq = np.zeros((len(crossover_rates), genome_length))
    two_point_freq = np.zeros((len(crossover_rates), genome_length))
    
    # Test each crossover method at each rate
    for rate_idx, rate in enumerate(crossover_rates):
        # Initialize crossover operators
        single_point_crossover = Crossover(crossover_rate=rate)
        two_point_crossover = Crossover(crossover_rate=rate)
        
        # Run trials for single-point crossover
        for _ in range(num_trials):
            # Select two random parents
            parent_indices = np.random.choice(population_size, 2, replace=False)
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
            
            # Perform single-point crossover and track crossover point
            _, _, point = single_point_crossover.single_point_crossover(parent1, parent2)
            if point is not None:
                single_point_freq[rate_idx, point] += 1
        
        # Run trials for two-point crossover
        for _ in range(num_trials):
            # Select two random parents
            parent_indices = np.random.choice(population_size, 2, replace=False)
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
            
            # Perform two-point crossover and track crossover points
            _, _, points = two_point_crossover.two_point_crossover(parent1, parent2)
            if points is not None:
                for point in points:
                    two_point_freq[rate_idx, point] += 1
    
    # Demonstrate a definite crossover example (random crossover points)
    def_crossover_rate = 1.0  # Ensure crossover occurs for demonstration
    def_single_crossover = Crossover(crossover_rate=def_crossover_rate)
    def_two_crossover = Crossover(crossover_rate=def_crossover_rate)
    
    # Use first two parents from population for demonstration
    demo_parent1 = population[0].copy()
    demo_parent2 = population[1].copy()
    
    # Perform single-point crossover
    demo_child1_single, demo_child2_single, single_point = def_single_crossover.single_point_crossover(demo_parent1, demo_parent2)
    
    # Perform two-point crossover
    demo_child1_two, demo_child2_two, two_points = def_two_crossover.two_point_crossover(demo_parent1, demo_parent2)
    
    # Create a figure with four subplots for the original visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    plt.subplots_adjust(hspace=0.50, wspace=0.2)
    
    # Plot 1: Original Population (Colored Grid)
    axes[0].set_title('Original Population')
    sns.heatmap(population, ax=axes[0], cmap='binary', cbar=False, annot=True, fmt='d')
    axes[0].set_ylabel('Individuals')
    axes[0].set_xlabel('Genome Position')
    
    # Plot 2: Single-Point Crossover Frequency
    axes[1].set_title('Single-Point Crossover Frequency Across Rates')
    sns.heatmap(single_point_freq, ax=axes[1], cmap='Blues', annot=True, fmt='.0f')
    axes[1].set_yticks(np.arange(len(crossover_rates)) + 0.5)
    axes[1].set_yticklabels([f'Rate {rate}' for rate in crossover_rates], rotation=0)
    axes[1].set_ylabel('Crossover Rate')
    axes[1].set_xlabel('Genome Position')
    
    # Plot 3: Two-Point Crossover Frequency
    axes[2].set_title('Two-Point Crossover Frequency Across Rates')
    sns.heatmap(two_point_freq, ax=axes[2], cmap='Reds', annot=True, fmt='.0f')
    axes[2].set_yticks(np.arange(len(crossover_rates)) + 0.5)
    axes[2].set_yticklabels([f'Rate {rate}' for rate in crossover_rates], rotation=0)
    axes[2].set_ylabel('Crossover Rate')
    axes[2].set_xlabel('Genome Position')
     
    # Adjust layout and save the first plot
    # plt.tight_layout()
    plt.show()
    plt.savefig('crossover_visualization.png')
    plt.close()
    
    # New Plot: Specific Crossover Example with Fixed Points
    # Define specific crossover points
    specific_single_point = 10  # Crossover at position 10 for single-point
    specific_two_points = [5, 15]  # Crossover at positions 5 and 15 for two-point
    
    # Create two specific parent genomes for clarity
    specific_parent1 = np.array([1] * 10 + [0] * 10)  # First half 1s, second half 0s
    specific_parent2 = np.array([0] * 10 + [1] * 10)  # First half 0s, second half 1s
    
    # Apply crossovers with specific points
    specific_single_crossover = Crossover(crossover_rate=1.0)
    specific_two_crossover = Crossover(crossover_rate=1.0)
    
    # Perform single-point crossover at specific point
    spec_child1_single, spec_child2_single, _ = specific_single_crossover.single_point_crossover(
        specific_parent1, specific_parent2, point=specific_single_point
    )
    
    # Perform two-point crossover at specific points
    spec_child1_two, spec_child2_two, _ = specific_two_crossover.two_point_crossover(
        specific_parent1, specific_parent2, points=specific_two_points
    )
    
    # Create a new figure for the specific crossover example
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Stack parents and children vertically for visualization
    specific_data = np.vstack((specific_parent1, specific_parent2, 
                               spec_child1_single, spec_child2_single, 
                               spec_child1_two, spec_child2_two))
    sns.heatmap(specific_data, ax=ax, cmap='binary', cbar=False, annot=True, fmt='d')
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_yticklabels(['Parent 1', 'Parent 2', 'Child 1 (Single)', 'Child 2 (Single)', 
                        'Child 1 (Two)', 'Child 2 (Two)'], rotation=0)
    ax.set_xlabel('Genome Position')
    ax.set_title('Specific Crossover Example with Fixed Points')
    # Mark crossover points
    ax.axvline(x=specific_single_point + 0.5, color='blue', linestyle='--', label='Single Point (Pos 10)')
    for point in specific_two_points:
        ax.axvline(x=point + 0.5, color='red', linestyle='--', label='Two Points (Pos 5, 15)' if point == specific_two_points[0] else "")
    ax.legend()
    
    # Save the specific crossover plot
    plt.tight_layout()
    plt.show()
    plt.savefig('specific_crossover_example.png')
    plt.close()
    
    # Print summary statistics for the frequency visualization
    summary = "=== Crossover Visualization Results ===\n\n"
    summary += "Single-Point Crossover - Total Crossovers per Rate:\n"
    for rate_idx, rate in enumerate(crossover_rates):
        total_crossovers = np.sum(single_point_freq[rate_idx])
        summary += f"Rate {rate}: {int(total_crossovers)} crossovers (Expected ~{int(rate * num_trials)})\n"
    
    summary += "\nTwo-Point Crossover - Total Crossovers per Rate:\n"
    for rate_idx, rate in enumerate(crossover_rates):
        total_crossovers = np.sum(two_point_freq[rate_idx])
        summary += f"Rate {rate}: {int(total_crossovers)} crossovers (Expected ~{int(2 * rate * num_trials)})\n"
    
    summary += "\nPlots saved as 'crossover_visualization.png' and 'specific_crossover_example.png'\n"
    
    return {
        'summary': summary,
        'population': population.tolist(),
        'single_point_freq': single_point_freq.tolist(),
        'two_point_freq': two_point_freq.tolist(),
        'crossover_rate': crossover_rate,
        'genome_length': genome_length,
        'population_size': population_size,
        'specific_single_point': specific_single_point,
        'specific_two_points': specific_two_points,
        'specific_data': specific_data.tolist()
    }