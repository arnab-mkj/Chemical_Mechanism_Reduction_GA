import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from matplotlib.colors import ListedColormap

class Crossover:
    def __init__(self, crossover_rate):
        self.crossover_rate = crossover_rate
        
    def single_point_crossover(self, parent1, parent2):
        """Perform single-point crossover between two parents with given probability"""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1)-1)  # Avoid trivial splits
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2, point  # Return crossover point
        return parent1.copy(), parent2.copy(), -1  # -1 indicates no crossover
    
    def two_point_crossover(self, parent1, parent2):
        """Perform two-point crossover between two parents with given probability"""
        if np.random.rand() < self.crossover_rate:
            points = sorted(np.random.choice(len(parent1)-1, 2, replace=False))
            child1 = np.concatenate((
                parent1[:points[0]], 
                parent2[points[0]:points[1]], 
                parent1[points[1]:]
            ))
            child2 = np.concatenate((
                parent2[:points[0]], 
                parent1[points[0]:points[1]], 
                parent2[points[1]:]
            ))
            return child1, child2, points  # Return crossover points
        return parent1.copy(), parent2.copy(), []  # Empty list indicates no crossover

def test_h2o2_crossover():
    # Parameters
    num_reactions = 29  # Number of reactions in the H2O2 mechanism
    population_size = 20  # Number of genomes in the population (smaller for visualization)
    crossover_rates = [0.3, 0.6, 0.9]  # Different crossover rates to test
    
    # Custom colormap for better visualization
    cmap = ListedColormap(['white', 'black'])  # 0=white, 1=black

    # Initialize population (random binary genomes)
    population = [np.random.choice([0, 1], size=num_reactions) for _ in range(population_size)]

    # Prepare data for saving to CSV
    results_data = []

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # Set main title
    fig.suptitle("Population-Level Crossover Analysis", y=1.02, fontsize=14)

    for col, rate in enumerate(crossover_rates):
        # Initialize crossover object
        crossover = Crossover(crossover_rate=rate)
        
        # Create copies of population for crossover
        shuffled_pop = population.copy()
        np.random.shuffle(shuffled_pop)
        
        # Apply crossover to population pairs
        new_population = []
        crossover_stats = []
        
        for i in range(0, len(shuffled_pop)-1, 2):
            parent1, parent2 = shuffled_pop[i], shuffled_pop[i+1]
            
            # Apply both crossover types
            sp_child1, sp_child2, sp_point = crossover.single_point_crossover(parent1, parent2)
            tp_child1, tp_child2, tp_points = crossover.two_point_crossover(parent1, parent2)
            
            # Store results
            new_population.extend([sp_child1, sp_child2, tp_child1, tp_child2])
            crossover_stats.append({
                'parents': (i, i+1),
                'sp_point': sp_point,
                'tp_points': tp_points,
                'did_crossover': sp_point != -1  # Same for both types in this test
            })

        # Calculate crossover statistics
        total_pairs = len(crossover_stats)
        crossover_count = sum(1 for s in crossover_stats if s['did_crossover'])
        crossover_percent = (crossover_count / total_pairs) * 100
        
        # Create population visualization
        ax1 = fig.add_subplot(gs[0, col])
        ax2 = fig.add_subplot(gs[1, col])
        ax3 = fig.add_subplot(gs[2, col])
        
        # Original population
        ax1.imshow(population, aspect="auto", cmap=cmap, interpolation="nearest")
        ax1.set_title(f"Original Population\nCrossover Rate: {rate}", pad=15)
        ax1.set_ylabel("Individuals")
        ax1.set_xlabel("Reactions")
        ax1.set_xticks(np.arange(num_reactions))
        ax1.set_xticklabels(np.arange(1, num_reactions+1), rotation=90, fontsize=8)
        
        # Single-point crossover results
        sp_pop = [child for i, child in enumerate(new_population) if i % 4 in [0,1]]
        ax2.imshow(sp_pop, aspect="auto", cmap=cmap, interpolation="nearest")
        ax2.set_title(f"Single-Point\n({crossover_count}/{total_pairs} pairs crossed)", pad=15)
        ax2.set_ylabel("Children")
        ax2.set_xlabel("Reactions")
        
        # Mark crossover points
        for row, stats in enumerate(crossover_stats):
            if stats['sp_point'] != -1:
                ax2.axvline(x=stats['sp_point']-0.5, color='red', linestyle='-', linewidth=0.5)
                ax2.plot(stats['sp_point'], row*2+0.5, 'ro', markersize=4)
        
        # Two-point crossover results
        tp_pop = [child for i, child in enumerate(new_population) if i % 4 in [2,3]]
        ax3.imshow(tp_pop, aspect="auto", cmap=cmap, interpolation="nearest")
        ax3.set_title(f"Two-Point\n({crossover_count}/{total_pairs} pairs crossed)", pad=15)
        ax3.set_ylabel("Children")
        ax3.set_xlabel("Reactions")
        
        # Mark crossover regions
        for row, stats in enumerate(crossover_stats):
            if stats['tp_points']:
                start, end = stats['tp_points']
                ax3.axvspan(start-0.5, end-0.5, facecolor='red', alpha=0.2)
                ax3.plot([start, end], [row*2+0.5, row*2+0.5], 'r-', linewidth=2)

        # Save data to results
        results_data.append({
            "Crossover Rate": rate,
            "Total Pairs": total_pairs,
            "Crossover Count": crossover_count,
            "Crossover Percentage": crossover_percent,
            "Example SP Point": next(s['sp_point'] for s in crossover_stats if s['sp_point'] != -1),
            "Example TP Points": next(str(s['tp_points']) for s in crossover_stats if s['tp_points'])
        })

    plt.tight_layout()
    plt.show()

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv("population_crossover_results.csv", index=False)
    print("Results saved to 'population_crossover_results.csv'")

if __name__ == "__main__":
    test_h2o2_crossover()