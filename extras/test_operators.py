import sys
import os
import numpy as np
from typing import List, Tuple

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ChemicalMechanismGA.operators.selection import Selection
from ChemicalMechanismGA.operators.mutation import Mutation
from ChemicalMechanismGA.operators.crossover import Crossover

def generate_population(pop_size: int, genome_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random population with corresponding fitness scores."""
    population = np.random.randint(0, 2, size=(pop_size, genome_length))
    fitness_scores = np.random.uniform(0.1, 10.0, size=pop_size)
    return population, fitness_scores

def print_population(population: np.ndarray, fitness_scores: np.ndarray) -> None:
    """Print population with fitness scores."""
    print("\nCurrent Population:")
    for i, genome in enumerate(population):
        print(f"Individual {i:02d}: {genome} (fitness: {fitness_scores[i]:.2f})")

def test_selection(pop_sizes: List[int] = [5, 10, 20], genome_lengths: List[int] = [8, 16, 32]):
    print("\n################ Testing Fitness Selection Operator ################")
    selection = Selection()
    
    for pop_size in pop_sizes:
        for genome_length in genome_lengths:
            print(f"\n=== Population Size: {pop_size}, Genome Length: {genome_length} ===")
            population, fitness_scores = generate_population(pop_size, genome_length)
            print_population(population, fitness_scores)
            
            print("\nTesting tournament selection:")
            for i in range(5):
                selected_idx = selection.tournament_selection(fitness_scores)
                print(f"Trial {i+1}: Selected index {selected_idx} with fitness {fitness_scores[selected_idx]:.2f}")

def test_crossover(pop_sizes: List[int] = [4, 8, 16], genome_lengths: List[int] = [8, 16, 32]):
    print("\n################ Testing Crossover Operator ################")
    
    for pop_size in pop_sizes:
        for genome_length in genome_lengths:
            print(f"\n=== Population Size: {pop_size}, Genome Length: {genome_length} ===")
            population, _ = generate_population(pop_size, genome_length)
            
            for crossover_rate in [0.2, 0.5, 0.8]:
                crossover = Crossover(crossover_rate=crossover_rate)
                print(f"\nCrossover Rate: {crossover_rate}")
                
                parent1, parent2 = population[:2]
                print(f"Parent 1: {parent1}")
                print(f"Parent 2: {parent2}")
                
                # Test single point crossover
                child1, child2 = crossover.single_point_crossover(parent1, parent2)
                print(f"\nSingle-point crossover results:")
                print(f"Child 1: {child1}")
                print(f"Child 2: {child2}")
                
                # Test two point crossover
                child1, child2 = crossover.two_point_crossover(parent1, parent2)
                print(f"\nTwo-point crossover results:")
                print(f"Child 1: {child1}")
                print(f"Child 2: {child2}")

def test_mutation(pop_sizes: List[int] = [4, 8], genome_lengths: List[int] = [8, 16, 32]):
    print("\n################ Testing Mutation Operator ################")
    
    for pop_size in pop_sizes:
        for genome_length in genome_lengths:
            print(f"\n=== Population Size: {pop_size}, Genome Length: {genome_length} ===")
            population, _ = generate_population(pop_size, genome_length)
            original_genome = population[0]
            print(f"Original genome: {original_genome}")
            
            for mutation_rate in [0.1, 0.5, 1.0]:
                mutation = Mutation(mutation_rate=mutation_rate)
                print(f"\nMutation Rate: {mutation_rate}")
                
                # Test bit flip mutation
                mutated_genome = mutation.bit_flip_mutation(original_genome.copy())
                differences = np.sum(original_genome != mutated_genome)
                print(f"Bit flip mutation: {mutated_genome}")
                print(f"Number of mutations: {differences}")
                
                # Test swap position mutation
                mutated_genome = mutation.swap_pos_mutation(original_genome.copy())
                differences = np.sum(original_genome != mutated_genome)
                print(f"Swap position mutation: {mutated_genome}")
                print(f"Number of mutations: {differences}")

def test_full_cycle(pop_sizes: List[int] = [10, 20], genome_lengths: List[int] = [16, 32]):
    print("\n################ Testing Full GA Cycle ################")
    
    selection = Selection()
    
    for pop_size in pop_sizes:
        for genome_length in genome_lengths:
            print(f"\n=== Population Size: {pop_size}, Genome Length: {genome_length} ===")
            
            for crossover_rate in [0.5, 0.8]:
                for mutation_rate in [0.1, 0.3]:
                    print(f"\nCrossover Rate: {crossover_rate}, Mutation Rate: {mutation_rate}")
                    crossover = Crossover(crossover_rate=crossover_rate)
                    mutation = Mutation(mutation_rate=mutation_rate)
                    
                    population, fitness_scores = generate_population(pop_size, genome_length)
                    print_population(population, fitness_scores)
                    
                    # Run multiple generations
                    for generation in range(3):
                        print(f"\n--- Generation {generation + 1} ---")
                        
                        new_population = []
                        for _ in range(pop_size // 2):
                            # Selection
                            parent1_idx = selection.tournament_selection(fitness_scores)
                            parent2_idx = selection.tournament_selection(fitness_scores)
                            
                            # Crossover
                            child1, child2 = crossover.single_point_crossover(
                                population[parent1_idx], 
                                population[parent2_idx]
                            )
                            
                            # Mutation
                            child1 = mutation.bit_flip_mutation(child1)
                            child2 = mutation.bit_flip_mutation(child2)
                            
                            new_population.extend([child1, child2])
                        
                        population = np.array(new_population)
                        fitness_scores = np.random.uniform(0.1, 10.0, size=pop_size)
                        print_population(population, fitness_scores)

def main():
    #np.random.seed(42)  # For reproducibility
    
    # Test individual components with various sizes
    test_selection(pop_sizes=[5, 10, 20], genome_lengths=[8, 16, 32])
    test_crossover(pop_sizes=[4, 8], genome_lengths=[8, 16, 32])
    test_mutation(pop_sizes=[4, 8], genome_lengths=[8, 16, 32])
    
    # Test full cycle with different parameters
    test_full_cycle(pop_sizes=[10, 20], genome_lengths=[16, 32])

if __name__ == "__main__":
    main()