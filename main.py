from src.ChemicalMechanismGA.components.genetic_algorithm import GeneticAlgorithm
from src.ChemicalMechanismGA.components.fitness_function import evaluate_fitness
#from src.ChemicalMechanismGA.utils.hyperparameter_tuning import HyperparameterTuner
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
import json


def main():
    # Step 1: Initialize GA parameters
    population_size = 50
    genome_length = 325  # Number of reactions in GRI-Mech 3.0
    crossover_rate = 0.8
    mutation_rate = 0.01
    num_generations = 500
    elite_size = 2
    
    # Paths for the mechanism files
    original_mechanism_path = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/gri30.yaml" 
    output_directory = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/output"  
    
    # Reactor type
    reactor_type = "constant_pressure"  # Example: batch reactor
    
    # Step 2: Initialize hyperparameter tuning
    # Example simulation results for hyperparameter
    results = {
        "temperature": 1950.0,
        "mole_fractions": {"CO2": 0.09, "H2O": 0.18},
        "ignition_delay": 1.2
    }
    
    #Define the Hyperparameter ranges
    """
    param_grid = {
        "weight_temperature": [0.5, 1.0, 1.5],
        "weight_species": [0.5, 1.0, 1.5],
        "weight_ignition_delay": [0.5, 1.0, 1.5],
        "difference_function": ["absolute", "squared", "relative"]
    }

    # Perform grid search for hyperparameter tuning
    tuner = HyperparameterTuner(FitnessEvaluator, results)
    best_hyperparams = tuner.grid_search(param_grid)
    print(f"Best Hyperparameters: {best_hyperparams['best_params']}")
    print(f"Best Fitness Score: {best_hyperparams['best_fitness']}")

    # Use the best hyperparameters in the fitness evaluator
    best_params = best_hyperparams["best_params"]
    """
    # Step 2: Initialize FitnessEvaluator with default parameters
    fitness_evaluator = FitnessEvaluator(
        target_temperature=2000.0,  # Example target temperature
        target_species={"CO2": 0.1, "H2O": 0.2},  # Example target species mole fractions
        target_delay=1.0,  # Example target ignition delay
        weight_temperature=1.0,  # Default weight for temperature fitness
        weight_species=1.0,  # Default weight for species fitness
        weight_ignition_delay=1.0,  # Default weight for ignition delay fitness
        difference_function="logarithmic",  # Default difference function
        sharpening_factor = 10.0 # empirical value
    )
    
    # Step 3: Create GA instance
    ga = GeneticAlgorithm(
        population_size=population_size,
        genome_length=genome_length,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        num_generations=num_generations,
        elite_size=elite_size
    )

    # Step 4: Run the GEnetic Algorithm
    best_genome, best_fitness = ga.evolve(fitness_function=evaluate_fitness,
                                          original_mechanism_path=original_mechanism_path,
                                          output_directory=output_directory,
                                          reactor_type=reactor_type) 
    #gets the fitness score from fitness_function.py
                   # and then passes to evolve in Genetic algo which then again passes to
                   #evaluate_fitness in population to store them in an array for a particular population
    # Step 5: Save results
    print(f"\nBest solution found:")
    print(f"Fitness: {best_fitness}")
    print(f"Number of active reactions: {sum(best_genome)}")
    
    
    
    # Save the best genome as a reduced mechanism in YAML format
    output_path = f"{output_directory}/best_reduced_mechanism.yaml"
    save_genome_as_yaml(best_genome, original_mechanism_path, output_path)
    print(f"Best reduced mechanism saved to: {output_path}")

    # Save fitness history for analysis
    fitness_history_path = f"{output_directory}/fitness_history.json"
    with open(fitness_history_path, "w") as file:
        json.dump(ga.fitness_history, file, indent=4)
    print(f"Fitness history saved to: {fitness_history_path}")
    
    # Save species concentrations
    species_concentrations_path = f"{output_directory}/mole_fractions.json"
    print(f"Species concentrations saved to: {species_concentrations_path}")
    
    
    
if __name__ == "__main__":
    main()