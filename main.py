from src.ChemicalMechanismGA.components.genetic_algorithm import GeneticAlgorithm
from src.ChemicalMechanismGA.components.fitness_function import evaluate_fitness

def main():
    # Initialize GA parameters
    population_size = 50
    genome_length = 325  # Number of reactions in GRI-Mech 3.0
    crossover_rate = 0.8
    mutation_rate = 0.1
    num_generations = 100

    # Create GA instance
    ga = GeneticAlgorithm(
        population_size=population_size,
        genome_length=genome_length,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        num_generations=num_generations
    )

    # Run evolution
    best_genome, best_fitness = ga.evolve(evaluate_fitness)

    print(f"\nBest solution found:")
    print(f"Fitness: {best_fitness}")
    print(f"Number of active reactions: {sum(best_genome)}")

if __name__ == "__main__":
    main()