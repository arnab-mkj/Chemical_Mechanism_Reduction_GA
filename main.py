from src.ChemicalMechanismGA.components.genetic_algorithm import GeneticAlgorithm
#from src.ChemicalMechanismGA.utils.hyperparameter_tuning import HyperparameterTuner
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
from src.ChemicalMechanismGA.components.population import Population
from src.ChemicalMechanismGA.components.simulation_runner import SimulationRunner
import json
import cantera as ct 

"""
    Todo: 
          > Improve the visualization to plot lower values
          > implement the burning velocity in PREMIX(optional)
          > Implement sensitivity analysis
          > Implement the third-body reaction mechanism (not needed, done by cantera)
          > Mark definite species and their corresponding reaction
"""
def main():
    # Step 1: Initialize GA parameters
    population_size = 50
    genome_length = 325  # Number of reactions in GRI-Mech 3.0
    crossover_rate = 0.6
    mutation_rate = 0.1
    num_generations = 500
    elite_size = 2
    
    # Paths for the mechanism files
    original_mechanism_path = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/gri30.yaml" 
    output_directory = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/output"  
    
    # Reactor type
    reactor_type = "constant_pressure"  # Example: batch reactor
  
    weights = {"temperature": 1, "species": 1, "ignition_delay": 1}
    #weights = {"temperature": 0.5, "species": 0.5}
    
    difference_function = "logarithmic"  # Default difference function
    sharpening_factor = 6 # empirical value
    lam = 0.1
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
            'fuel': {'CH4'}, #mole fraction
            'oxidizer': {'O2': 0.1938, 'N2': 0.7287},
            'pressure': 2670 * ct.one_atm / 101325, #* 0.0263, (20 torr)
            'temperature': 1800.0,
            'mass_flow_rate': 0.0989 #kg/m^2/s
        }
    species_def = ['CH4','O2','CO2','H2O','CO','H2','O','OH','H','CH3']
    fitness_evaluator = FitnessEvaluator(
        original_mechanism_path,
        reactor_type,
        condition,
        weights,
        difference_function,
        sharpening_factor,
        lam
        )
  
    # Step 3: Create GA instance
    ga = GeneticAlgorithm(
        population_size,
        genome_length,
        crossover_rate,
        mutation_rate,
        num_generations,
        elite_size,
        fitness_evaluator,
        species_def# ????? needed?? check!!!!
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