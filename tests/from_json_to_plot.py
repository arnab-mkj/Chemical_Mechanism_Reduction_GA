import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_generation_data(json_file):
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize lists to store data
    generations = []
    best_fitness_values = []
    filtered_generations = []
    filtered_fitness_values = []
    active_reactions_counts = []
    filtered_active_reactions = []
    best_genomes = []

    # Extract data for each generation
    for entry in data:
        generation = entry["generation"]
        best_fitness = entry["best_fitness"]
        best_genome = entry["overall_best_genome"]

        # Count active reactions in the genome (sum of 1s)
        active_reactions = sum(best_genome)

        generations.append(generation)
        best_fitness_values.append(best_fitness)
        active_reactions_counts.append(active_reactions)
        best_genomes.append(best_genome)

        # Filter values below 2*10^-21
        if best_fitness < (2 ):
            filtered_generations.append(generation)
            filtered_fitness_values.append(best_fitness)
            filtered_active_reactions.append(active_reactions)

    # Find the overall best fitness and its generation
    if filtered_fitness_values:
        overall_best_fitness = min(filtered_fitness_values)
        best_fitness_index = filtered_fitness_values.index(overall_best_fitness)
        best_generation = filtered_generations[best_fitness_index]
        best_active_reactions = filtered_active_reactions[best_fitness_index]

        # Find the corresponding genome
        original_index = generations.index(best_generation)
        best_genome = best_genomes[original_index]

        print(f"Overall Best Fitness: {overall_best_fitness} (Generation {best_generation})")
        print(f"Number of active reactions in best genome: {best_active_reactions}")
        print(f"Generations with Fitness < 2*10^-21: {filtered_generations}")
        print(f"Fitness Values: {filtered_fitness_values}")

        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Best Fitness vs. Generations (Filtered)
        ax1.plot(filtered_generations, filtered_fitness_values, marker='o', color='blue', label="Best Fitness (< 2*10^-21)")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Best Fitness")
        ax1.set_title("Best Fitness vs. Generations ")
        ax1.set_yscale("log")  # Log scale for better visualization
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Active Reactions vs. Generations
        ax2.plot(generations, active_reactions_counts, marker='s', color='green', label="Active Reactions")
        # Highlight the best genome point
        ax2.scatter([best_generation], [best_active_reactions], color='red', s=100,
                   label=f"Best Genome ({best_active_reactions} reactions)")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Number of Active Reactions")
        ax2.set_title("Number of Active Reactions vs. Generations")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Save the best genome to a file
        with open("best_genome.json", "w") as f:
            json.dump({
                "generation": best_generation,
                "fitness": overall_best_fitness,
                "active_reactions": best_active_reactions,
                "genome": best_genome
            }, f, indent=4)

        print(f"Best genome saved to 'best_genome.json'")

        return best_genome
    else:
        print("No fitness values below the threshold were found.")
        return None

# Example usage
json_file = "results/generation_stats.json"  # Replace with your JSON file name
best_genome = analyze_generation_data(json_file)