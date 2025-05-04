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

def test_h2o2_mutation(mutation_rate):
    # Parameters
    num_reactions = 29  # Number of reactions in the H2O2 mechanism
    population_size = 50  # Number of genomes in the population

    # Initialize population (random binary genomes)
    population = [np.random.choice([0, 1], size=num_reactions) for _ in range(population_size)]

    # Prepare data for saving to CSV
    results_data = []

    # Visualization for the given mutation rate
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.4)

    # Initialize mutation object
    mutation = Mutation(mutation_rate=mutation_rate)

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
            "Mutation Rate": mutation_rate,
            "Genome Index": i,
            "Initial Genome": "".join(map(str, genome)),
            "Bit Flip Genome": "".join(map(str, bit_flip_population[i])),
            "Swap Pos Genome": "".join(map(str, swap_pos_population[i])),
            "Bit Flip Changes": np.sum(bit_flip_changes[i]),
            "Swap Pos Changes": np.sum(swap_pos_changes[i]),
        })

    # Visualization
    axes[0].imshow(population, aspect="auto", cmap="binary", interpolation="nearest")
    axes[0].set_title(f"Initial Population")
    axes[0].set_xlabel("Reaction Index")
    axes[0].set_ylabel("Genome (Population)")

    axes[1].imshow(bit_flip_population, aspect="auto", cmap="binary", interpolation="nearest")
    axes[1].set_title(f"Bit Flip Mutation (Rate={mutation_rate})")
    axes[1].set_xlabel("Reaction Index")
    axes[1].set_ylabel("Genome (Population)")

    for row, changes in enumerate(bit_flip_changes):
        for idx, changed in enumerate(changes):
            if changed:
                rect = patches.Rectangle((idx - 0.5, row - 0.5), 1, 1, linewidth=0.8, edgecolor="red", facecolor="none")
                axes[1].add_patch(rect)

    axes[2].imshow(swap_pos_population, aspect="auto", cmap="binary", interpolation="nearest")
    axes[2].set_title(f"Swap Position Mutation (Rate={mutation_rate})")
    axes[2].set_xlabel("Reaction Index")
    axes[2].set_ylabel("Genome (Population)")

    for row, changes in enumerate(swap_pos_changes):
        for idx, changed in enumerate(changes):
            if changed:
                rect = patches.Rectangle((idx - 0.5, row - 0.5), 1, 1, linewidth=0.8, edgecolor="red", facecolor="none")
                axes[2].add_patch(rect)
    plt.show()
    # plt.savefig("mutation_results.png")
    plt.close()

    results_df = pd.DataFrame(results_data)
    results_df.to_csv("mutation_results.csv", index=False)
    return "Mutation test completed. Results saved to 'mutation_results.csv' and 'mutation_results.png'"

def run_test(mutation_rate):
    # Since test_h2o2_mutation doesn't use the mechanism or config parameters, we can directly call it with the custom mutation rate
    result = test_h2o2_mutation(mutation_rate)
    return result

if __name__ == "__main__":
    test_h2o2_mutation(0.01)  # Example mutation rate