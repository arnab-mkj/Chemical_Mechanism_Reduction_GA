import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ChemicalMechanismGA.operators.selection import Selection
from ChemicalMechanismGA.operators.mutation import Mutation
from ChemicalMechanismGA.operators.crossover import Crossover

import numpy as np

def test_selection():
    print("\n ################Testing Fitness Selection Operator")
    selection = Selection()
    
    fitness_scores = np.array([10.0, 5.0, 8.0, 3.0, 7.0, 1.0, 9.0, 4.0, 2.0]) # lower is better
    print(f"Fitness Scores: {fitness_scores}")
    
    print("\n Testing tournament selection")
    for i in range(10):
        selected_idx = selection.tournament_selection(fitness_scores)
        print(f"Trial{i+1}: Selected index {selected_idx} with fitness {fitness_scores[selected_idx]}")


def test_crossover():
    print("\n ######## Testing Crossover Operator ########")
    crossover = Crossover(crossover_rate = 0.2)
    
    # Create sample parents
    parent1 = np.array([1,1,1,1,1,1,1])
    parent2 = np.array([0,0,0,0,0,0,0])
    print (f"Parent 1: {parent1} \n Parent 2: {parent2}")
    
    # test single point crossover
    print("\n Testing single point crossover :")
    for i in range(3):
        child1, child2 = crossover.single_point_crossover(parent1, parent2)
        print(f"\nTrial {i+1}")
        print(f"child 1: {child1}")
        print(f"child 2: {child2}")
        
    # test two point crossover
    print("\n Testing two point crossover :")
    for i in range(3):
        child1, child2 = crossover.two_point_crossover(parent1, parent2)
        print(f"\nTrial {i+1}")
        print(f"child 1: {child1}")
        print(f"child 2: {child2}")
    

def test_mutation():
    print("\n########## Testing Mutation Operator")
    mutation = Mutation(mutation_rate=0.8)
    
    original_genome = np.array([1,1,1,1,0,0,0,0])
    print(f"Priginal genome: {original_genome}")
    
    print("\n Test bit flip mutation")
    for i in range(3):
        mutated_genome = mutation.bit_flip_mutation(original_genome)
        print(f"Trial {i+1}: {mutated_genome}")
        differences = np.sum(original_genome != mutated_genome)
        print(f"Number of mutations(bit flip): {differences}")
        
    print("\n Test swap position mutation")
    for i in range(3):
        mutated_genome = mutation.swap_pos_mutation(original_genome)
        print(f"Trial {i+1}: {mutated_genome}")
        differences = np.sum(original_genome != mutated_genome)
        print(f"Number of mutations(swap position): {differences}")



def test_full_cycle():
    print("\n########### Test full cycle")
    
    selection = Selection()
    crossover = Crossover(crossover_rate=0.8)
    mutation = Mutation(mutation_rate=0.1)
    
    #Creating sample population and fitness scores
    population = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ])
    fitness_scores = np.array([5.0, 3.0, 8.0, 4.0])
    
    print("Inital Population:")
    for i, genome in enumerate(population):
        print(f"Individual {i}: {genome} (fitness: {fitness_scores[i]})")
        
    print("\nTesting complete cycle")
    
    # 1. Selection
    parent1_idx = selection.tournament_selection(fitness_scores)
    parent2_idx = selection.tournament_selection(fitness_scores)
    print(f"\nSelected parents indices: {parent1_idx}, {parent2_idx}")
    print(f"Parent 1: {population[parent1_idx]}")
    print(f"Parent 2: {population[parent2_idx]}")

    # 2. Crossover
    child1, child2 = crossover.single_point_crossover(population[parent1_idx], population[parent2_idx])
    print(f"\nAfter crossover:")
    print(f"Child 1: {child1}")
    print(f"Child 2: {child2}")

    # 3. Mutation
    mutated_child1 = mutation.bit_flip_mutation(child1)
    mutated_child2 = mutation.bit_flip_mutation(child2)
    print(f"\nAfter mutation:")
    print(f"Mutated Child 1: {mutated_child1}")
    print(f"Mutated Child 2: {mutated_child2}")
    pass

def main():
    # Set random seed for reproducibility
    #np.random.seed(42)

    # Run individual tests
    #test_selection()
    #test_crossover()
    #test_mutation()

    # Run complete cycle test
    test_full_cycle()

if __name__ == "__main__":
    main()