import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_generation_data(json_file, file):
    # Load the JSON file
    with open(f"{json_file}/{file}", 'r') as f:
        data = json.load(f)

    # Initialize lists to store data
    generations = []
    best_fitness_values = []
    mean_fitness_values = []
    temperature_fitness_min = []
    species_fitness_min = []
    ignition_delay_fitness_min = []
    reaction_fitness_min = []
    active_reactions_counts = []
    mean_active_reactions = []
    best_genomes = []

    # Extract data for each generation
    for entry in data:
        generation = entry["generation"]
        best_fitness = entry["best_fitness"]
        mean_fitness = entry["mean_fitness"]
        temp_fit_min = entry["temperature_fitness_min"]
        species_fit_min = entry["species_fitness_min"]
        ignition_fit_min = entry["ignition_delay_fitness_min"]
        reaction_fit_min = entry["reaction_fitness_min"]
        best_genome = entry["overall_best_genome"]

        # Count active reactions in the genome (sum of 1s)
        active_reactions = entry["best_reactions"]
        mean_reactions = entry["active_reactions_mean"]

        generations.append(generation)
        best_fitness_values.append(best_fitness)
        mean_fitness_values.append(mean_fitness)
        temperature_fitness_min.append(temp_fit_min)
        species_fitness_min.append(species_fit_min)
        ignition_delay_fitness_min.append(ignition_fit_min)
        reaction_fitness_min.append(reaction_fit_min)
        active_reactions_counts.append(active_reactions)
        mean_active_reactions.append(mean_reactions)
        best_genomes.append(best_genome)

    # Find the overall best fitness and its generation
    if best_fitness_values:
        overall_best_fitness = min(best_fitness_values)
        best_fitness_index = best_fitness_values.index(overall_best_fitness)
        best_generation = generations[best_fitness_index]
        best_active_reactions = active_reactions_counts[best_fitness_index]

        # Find the corresponding genome
        best_genome = best_genomes[best_fitness_index]

        print(f"Overall Best Fitness: {overall_best_fitness} (Generation {best_generation})")
        print(f"Number of active reactions in best genome: {best_active_reactions}")

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.50, wspace=0.2)

        # Plot 1: Best and Mean Fitness vs. Generations
        axs[0, 0].plot(generations, best_fitness_values, linestyle='-', color='blue', label="Best Fitness")
        axs[0, 0].plot(generations, mean_fitness_values, linestyle='--', color='orange', label="Mean Fitness")
        axs[0, 0].set_xlabel("Generation")
        axs[0, 0].set_ylabel("Fitness")
        axs[0, 0].set_title("Best and Mean Fitness vs. Generations")
        # axs[0, 0].set_yscale("log")  # Log scale for better visualization
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # Plot 2: Active Reactions vs. Generations
        axs[0, 1].plot(generations, active_reactions_counts, linestyle='-', color='green', label="Best Active Reactions")
        axs[0, 1].plot(generations, mean_active_reactions, linestyle='--', color='purple', label="Mean Active Reactions")
        axs[0, 1].scatter([best_generation], [best_active_reactions], color='red', s=100,
                          label=f"Best Genome ({best_active_reactions} reactions)")
        axs[0, 1].set_xlabel("Generation")
        axs[0, 1].set_ylabel("Number of Active Reactions")
        axs[0, 1].set_title("Number of Active Reactions vs. Generations")
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        # Plot 3: Temperature Fitness vs. Generations
        axs[1, 0].plot(generations, temperature_fitness_min, linestyle='-', color='red', label="Temperature Fitness")
        axs[1, 0].set_xlabel("Generation")
        axs[1, 0].set_ylabel("Fitness")
        axs[1, 0].set_title("Temperature Fitness vs. Generations")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # Plot 4: Species Fitness vs. Generations
        axs[1, 1].plot(generations, species_fitness_min, linestyle='-', color='cyan', label="Species Fitness")
        axs[1, 1].set_xlabel("Generation")
        axs[1, 1].set_ylabel("Fitness")
        axs[1, 1].set_title("Species Fitness vs. Generations")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        # Plot 5: Ignition Delay Fitness vs. Generations
        axs[2, 0].plot(generations, ignition_delay_fitness_min, linestyle='-', color='magenta', label="Ignition Delay Fitness")
        axs[2, 0].set_xlabel("Generation")
        axs[2, 0].set_ylabel("Fitness")
        axs[2, 0].set_title("Ignition Delay Fitness vs. Generations")
        axs[2, 0].grid(True)
        axs[2, 0].legend()

        # Plot 6: Reaction Fitness vs. Generations
        axs[2, 1].plot(generations, reaction_fitness_min, linestyle='-', color='brown', label="Reaction Fitness")
        axs[2, 1].set_xlabel("Generation")
        axs[2, 1].set_ylabel("Fitness")
        axs[2, 1].set_title("Reaction Fitness vs. Generations")
        axs[2, 1].grid(True)
        axs[2, 1].legend()

        # plt.tight_layout()
        plt.show()

        # Save the best genome to a file
        with open(f"{json_file}/best_genome.json", "w") as f:
            json.dump({
                "generation": best_generation,
                "fitness": overall_best_fitness,
                "active_reactions": best_active_reactions,
                "genome": best_genome
            }, f, indent=4)

        print(f"Best genome saved to 'best_genome.json'")

        return best_genome
    else:
        print("No fitness values were found.")
        return None

# Example usage
json_file = "E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/absolute"  # Replace with your JSON file name
best_genome = analyze_generation_data(json_file, file="generation_stats.json")