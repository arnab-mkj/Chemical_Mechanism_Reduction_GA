import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

class Mutation:
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def bit_flip_mutation(self, genome):
        """Flip bits in the genome with a probability equal to the mutation rate."""
        for i in range(len(genome)):
            if np.random.rand() < self.mutation_rate:
                genome[i] = 1 - genome[i]
        return genome

    def swap_pos_mutation(self, genome):
        """Swap two positions in the genome with a probability equal to the mutation rate."""
        mutated_genome = genome.copy()
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(len(genome), 2, replace=False)
            mutated_genome[idx1], mutated_genome[idx2] = mutated_genome[idx2], mutated_genome[idx1]
        return mutated_genome


# Test Case for H2O2 Mechanism
def test_h2o2_mutation():
    # Parameters
    num_reactions = 29  # Number of reactions in the H2O2 mechanism
    population_size = 50  # Number of genomes in the population
    mutation_rates = [0.001, 0.01, 0.1]  # Different mutation rates to test

    # Initialize population (random binary genomes)
    population = [np.random.choice([0, 1], size=num_reactions) for _ in range(population_size)]

    # Prepare data for saving to CSV
    results_data = []

    # Visualization for each mutation rate
    fig, axes = plt.subplots(3, len(mutation_rates), figsize=(15, 12))
    plt.subplots_adjust(hspace=0.50, wspace=0.2)
    fig.suptitle("Effect of Mutation on H2O2 Mechanism", fontsize=16)

    for col, rate in enumerate(mutation_rates):
        # Initialize mutation object
        mutation = Mutation(mutation_rate=rate)

        # Apply bit flip mutation
        bit_flip_population = []
        bit_flip_changes = []
        for genome in population:
            mutated_genome = mutation.bit_flip_mutation(genome.copy())
            bit_flip_population.append(mutated_genome)
            bit_flip_changes.append(mutated_genome != genome)  # Track changed positions

        # Apply swap position mutation
        swap_pos_population = []
        swap_pos_changes = []
        for genome in population:
            mutated_genome = mutation.swap_pos_mutation(genome.copy())
            swap_pos_population.append(mutated_genome)
            swap_pos_changes.append(mutated_genome != genome)  # Track changed positions

        # Save data to results
        for i, genome in enumerate(population):
            results_data.append({
                "Mutation Rate": rate,
                "Genome Index": i,
                "Initial Genome": "".join(map(str, genome)),
                "Bit Flip Genome": "".join(map(str, bit_flip_population[i])),
                "Swap Pos Genome": "".join(map(str, swap_pos_population[i])),
                "Bit Flip Changes": np.sum(bit_flip_changes[i]),
                "Swap Pos Changes": np.sum(swap_pos_changes[i]),
            })

        # Visualization
        # Initial Population
        # if col == 0:  # Only plot the initial population once in the first column
        axes[0, col].imshow(population, aspect="auto", cmap="binary", interpolation="nearest")
        axes[0, col].set_title(f"Initial Population")
        axes[0, col].set_xlabel("Reaction Index")
        axes[0, col].set_ylabel("Genome (Population)")

        # Bit Flip Mutation
        axes[1, col].imshow(bit_flip_population, aspect="auto", cmap="binary", interpolation="nearest")
        axes[1, col].set_title(f"Bit Flip Mutation (Rate={rate})")
        axes[1, col].set_xlabel("Reaction Index")
        axes[1, col].set_ylabel("Genome (Population)")

        # Mark changed positions for bit flip
        for row, changes in enumerate(bit_flip_changes):
            for idx, changed in enumerate(changes):
                if changed:
                    rect = patches.Rectangle((idx - 0.5, row - 0.5), 1, 1, linewidth=0.5, edgecolor="red", facecolor="none")
                    axes[1, col].add_patch(rect)

        # Swap Position Mutation
        axes[2, col].imshow(swap_pos_population, aspect="auto", cmap="binary", interpolation="nearest")
        axes[2, col].set_title(f"Swap Position Mutation (Rate={rate})")
        axes[2, col].set_xlabel("Reaction Index")
        axes[2, col].set_ylabel("Genome (Population)")

        # Mark changed positions for swap position
        for row, changes in enumerate(swap_pos_changes):
            for idx, changed in enumerate(changes):
                if changed:
                    rect = patches.Rectangle((idx - 0.5, row - 0.5), 1, 1, linewidth=0.5, edgecolor="blue", facecolor="none")
                    axes[2, col].add_patch(rect)

    # Adjust layout
    
    # plt.tight_layout()
    plt.show()

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv("mutation_results.csv", index=False)
    print("Results saved to 'mutation_results.csv'")


# Run the test case
if __name__ == "__main__":
    test_h2o2_mutation()