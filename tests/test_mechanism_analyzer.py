
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ChemicalMechanismGA.components.population import Population

population_size = 50
genome_length = 325  # Number of reactions in GRI-Mech 3.0
crossover_rate = 0.8
mutation_rate = 0.01
num_generations = 500
elite_size = 2


population = Population(population_size, genome_length)

initial_popu = population.initialize_population()
print(type(initial_popu))
print("Dimension of initial population:", len(initial_popu), "x", len(initial_popu[0]))

    
#     <class 'numpy.ndarray'>
# Dimension of initial population: 50 x 325