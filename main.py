import json
import cantera as ct
from src.ChemicalMechanismGA.components.genetic_algorithm import GeneticAlgorithm
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
import time

def main():
    # Step 1: Load configuration file
    start_time = time.time()
    config_path = "params.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Step 2: Load species weights from a separate file
    weights_file = config["weights"]["species"]
    with open(weights_file, "r") as f:
        species_weights = json.load(f)
    
    # Step 3: Extract parameters from the configuration
    population_size = config["population_size"]
    genome_length = config["genome_length"]
    crossover_rate = config["crossover_rate"]
    mutation_rate = config["mutation_rate"]
    num_generations = config["num_generations"]
    elite_size = config["elite_size"]
    original_mechanism_path = config["original_mechanism_path"]
    output_directory = config["output_directory"]
    reactor_type = config["reactor_type"]
    difference_function = config["difference_function"]
    sharpening_factor = config["sharpening_factor"]
    normalization_method = config["normalization_method"]
    conditions = config["conditions"]
    key_species = config["key_species"]
    weights = {
        "temperature": config["weights"]["temperature"],
        "IDT": config["weights"]["IDT"],
        "species": species_weights,
        "reactions": config["weights"] ["reactions"]
    }
    

    # Step 4: Initialize the fitness evaluator
    fitness_evaluator = FitnessEvaluator(
        original_mechanism_path,
        reactor_type,
        conditions,
        weights,
        genome_length,
        difference_function,
        sharpening_factor,
        normalization_method,
        key_species
    )

    # Step 5: Create GA instance
    ga = GeneticAlgorithm(
        population_size,
        genome_length,
        crossover_rate,
        mutation_rate,
        num_generations,
        elite_size,
        fitness_evaluator,
        key_species, 
        difference_function,
    )

    # Step 6: Run the Genetic Algorithm
    best_genome, best_fitness = ga.evolve(output_directory)

    # Step 7: Save results
    print(f"\nBest solution found:")
    print(f"Fitness: {best_fitness}")
    print(f"Number of active reactions: {sum(best_genome)}")

    # Save the best genome as a reduced mechanism in YAML format
    output_path = f"{output_directory}/{difference_function}"
    save_genome_as_yaml(best_genome, original_mechanism_path, output_path)
    print(f"Best reduced mechanism saved to: {output_path}")

    # Save fitness history for analysis
    fitness_history_path = f"{output_directory}/{difference_function}/fitness_history.json"
    with open(fitness_history_path, "w") as file:
        json.dump(ga.fitness_history, file, indent=4)
    print(f"Fitness history saved to: {fitness_history_path}")
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\nTotal runtime: {total_runtime:.2f} seconds")
    
if __name__ == "__main__":
    main()