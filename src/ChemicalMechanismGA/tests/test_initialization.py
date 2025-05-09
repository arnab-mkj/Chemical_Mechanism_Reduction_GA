import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

class GeneticAlgorithm:
    def __init__(self, genome_length, popu_size, reaction_equations):
        self.genome_length = genome_length  # Number of reactions
        self.popu_size = popu_size          # Population size
        self.reaction_equations = reaction_equations  # List of reaction equations

    def initialize_population_with_removal(self, diversity_prob, remove_indices=None, remove_reactions=None):
        # Base genome with all reactions active
        base_genome = np.ones(self.genome_length, dtype=int)

        # Manually deactivate reactions by indices
        if remove_indices:
            for idx in remove_indices:
                base_genome[idx] = 0

        # Manually deactivate reactions by equations
        if remove_reactions:
            for reaction in remove_reactions:
                try:
                    idx = self.reaction_equations.index(reaction)
                    base_genome[idx] = 0
                except ValueError:
                    print(f"Reaction '{reaction}' not found in the mechanism.")

        population = []
        for _ in range(self.popu_size):
            # Start with the base genome
            genome = base_genome.copy()
            if diversity_prob > 0.0:  # Only introduce diversity if the probability is greater than 0
                for i in range(self.genome_length):
                    if np.random.rand() < diversity_prob:  # Probability of deactivating a reaction
                        genome[i] = 0  # Deactivate the reaction
            population.append(genome)

        return np.array(population)

def run_test(diversity_prob, remove_reactions):
    # Load the H2O2 mechanism using Cantera
    gas = ct.Solution("h2o2.yaml")
    reaction_equations = gas.reaction_equations()  # Get all reaction equations

    # Parameters for the H2O2 mechanism
    genome_length = len(reaction_equations)
    popu_size = 20  # Population size

    # Create an instance of the GeneticAlgorithm class
    ga = GeneticAlgorithm(genome_length, popu_size, reaction_equations)

    # Use the single diversity probability provided
    population = ga.initialize_population_with_removal(diversity_prob)
    population_with_reactions = ga.initialize_population_with_removal(0.0, remove_reactions=remove_reactions)

    # Print detailed results for the diversity probability
    print(f"\nDiversity Probability: {diversity_prob*100:.0f}%")
    print(f"Population Size: {popu_size}, Genome Length: {genome_length}")
    active_counts = np.sum(population, axis=1)  # Count active reactions in each genome
    print("Active Reactions per Genome:", active_counts)
    print(f"Average Active Reactions: {np.mean(active_counts):.2f}")
    print(f"Minimum Active Reactions: {np.min(active_counts)}")
    print(f"Maximum Active Reactions: {np.max(active_counts)}")

    print("\nPopulation with reactions removed by equation:")
    print(population_with_reactions)

    # Visualization
    plt.figure(figsize=(10, 5))

    # Heatmap for diversity probability
    plt.subplot(1, 2, 1)
    plt.imshow(population, cmap='Greys', aspect='auto', vmin=0, vmax=1)
    plt.title(f"Diversity Prob: {diversity_prob*100:.0f}%")
    plt.xlabel("Reaction Index")
    plt.ylabel("Genome Index")
    plt.colorbar(label="Reaction State (1=Active, 0=Inactive)")

    # Heatmap for manual removal by equations
    plt.subplot(1, 2, 2)
    plt.imshow(population_with_reactions, cmap='Greys', aspect='auto', vmin=0, vmax=1)
    plt.title("Manual Removal by Equation")
    plt.xlabel("Reaction Index")
    plt.ylabel("Genome Index")
    plt.colorbar(label="Reaction State (1=Active, 0=Inactive)")

    plt.tight_layout()
    plt.show()
    plt.savefig("initialization_results.png")
    plt.close()

    return "Initialization test completed. Results saved to 'initialization_results.png'"

if __name__ == "__main__":
    run_test(diversity_prob=0.05, remove_reactions=["H + O2 <=> O + OH", "HO2 + OH <=> H2O + O2"])