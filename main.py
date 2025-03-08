from src.ChemicalMechanismGA.components.genetic_algorithm import GeneticAlgorithm
#from src.ChemicalMechanismGA.utils.hyperparameter_tuning import HyperparameterTuner
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
from src.ChemicalMechanismGA.components.population import Population
from src.ChemicalMechanismGA.components.simulation_runner import SimulationRunner
import json
import cantera as ct 


def main():
    # Step 1: Initialize GA parameters
    population_size = 30
    genome_length = 325  # Number of reactions in GRI-Mech 3.0
    crossover_rate = 0.6
    mutation_rate = 0.05
    num_generations = 500
    elite_size = 2
    
    # Paths for the mechanism files
    original_mechanism_path = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/gri30.yaml" 
    output_directory = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/output"  
    
    # Reactor type
    reactor_type = "PREMIX"  # Example: batch reactor
  
    target_species={"CO2":0.1, "H2O":0.1, "CH4":0.1, "O2":0.1, "N2":0.1} # Example target species mole fractions
    target_delay=1.0  # Example target ignition delay
    weight_temperature=1.0  # Default weight for temperature fitness
    weight_species=1.0  # Default weight for species fitness
    difference_function="logarithmic"  # Default difference function
    sharpening_factor = 10.0 # empirical value
    initial_temperature = 1000
    initial_pressure = ct.one_atm
    #region
    # conditions_list = [
    #     # Condition 1: Lean flame (phi=0.8)
    #     {
    #         'phi': 0.8,
    #         'fuel': {'CH4': 1.0},
    #         'oxidizer': {'O2': 0.21, 'N2': 0.79},
    #         'pressure': ct.one_atm,
    #         'temperature': 300.0,
    #
    #     },
    #     # Condition 2: Stoichiometric flame (phi=1.0)
    #     {
    #         'phi': 1.0,
    #         'fuel': {'CH4': 1.0},
    #         'oxidizer': {'O2': 0.21, 'N2': 0.79},
    #         'pressure': ct.one_atm,
    #         'temperature': 300.0
    #     },
    #     # Condition 3: Rich flame (phi=1.2)
    #     {
    #         'phi': 1.2,
    #         'fuel': {'CH4': 1.0},
    #         'oxidizer': {'O2': 0.21, 'N2': 0.79},
    #         'pressure': ct.one_atm,
    #         'temperature': 300.0
    #     }
    # ]
    
    # fitness = evaluate_reduction_mechanism('gri30.yaml', 'reduced_16sp.yaml', conditions_list)
    # print(f"Fitness of reduced mechanism: (fitness:.8f)")
    # higher value for better reduced mechanism
    #endregion
    condition = {
            'phi': 0.8,
            'fuel': {'CH4': 1.0},
            'oxidizer': {'O2': 0.21, 'N2': 0.79},
            'pressure': ct.one_atm,
            'temperature': 300.0,
            'mass_flow_rate': 0.04
        }
    
    fitness_evaluator = FitnessEvaluator(
        original_mechanism_path,
        reactor_type,
        condition,
        weight_species,
        difference_function,
        sharpening_factor
        )
    #region
    # population = Population(population_size, genome_length) # creating an instance of population
    # initial_popu = population.initialize_population()

    # fitness_score_init = fitness_evaluator.run_generation(initial_popu, 0)
    # print(fitness_score)
    
    
    # pass to evolve, evolve should pass to run gen
    # exit()
    #endregion
    # Step 3: Create GA instance
    ga = GeneticAlgorithm(
        population_size,
        genome_length,
        crossover_rate,
        mutation_rate,
        num_generations,
        elite_size,
        fitness_evaluator  # ????? needed?? check!!!!
    )

    # Step 4: Run the GEnetic Algorithm
    best_genome, best_fitness = ga.evolve(output_directory
                                          ) 
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