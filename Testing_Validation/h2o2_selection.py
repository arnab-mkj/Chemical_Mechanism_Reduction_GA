import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

class Selection:

    @staticmethod
    def tournament_selection(fitness_scores, tournament_size=5):

        indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        return indices[np.argmin(fitness_scores[indices])]  # Return the index of the best individual in the tournament

    @staticmethod
    def roulette_wheel_selection(fitness_scores):

        total_fit = sum(fitness_scores)
        prob = [f / total_fit for f in fitness_scores]
        return np.random.choice(len(fitness_scores), p=prob)

# Load the H2O2 mechanism using Cantera
gas = ct.Solution("h2o2.yaml")
reaction_equations = gas.reaction_equations()  # Get all reaction equations

# Parameters
population_size = 20  # Number of individuals in the population
fitness_scores = np.random.uniform(0.1, 1.0, population_size)  # Random fitness scores (lower is better)

# Test selection methods
tournament_selected_indices = []
roulette_selected_indices = []

# Perform multiple selections to observe trends
num_selections = 100
for _ in range(num_selections):
    tournament_selected_indices.append(Selection.tournament_selection(fitness_scores))
    roulette_selected_indices.append(Selection.roulette_wheel_selection(fitness_scores))

# Count the frequency of selection for each individual
tournament_selection_counts = np.bincount(tournament_selected_indices, minlength=population_size)
roulette_selection_counts = np.bincount(roulette_selected_indices, minlength=population_size)

# Visualization
x = np.arange(population_size)

plt.figure(figsize=(12, 6))

# Tournament Selection
plt.subplot(1, 2, 1)
plt.bar(x, tournament_selection_counts, color='blue', alpha=0.7)
plt.xlabel("Individual Index")
plt.ylabel("Selection Count")
plt.title("Tournament Selection")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Roulette Wheel Selection
plt.subplot(1, 2, 2)
plt.bar(x, roulette_selection_counts, color='green', alpha=0.7)
plt.xlabel("Individual Index")
plt.ylabel("Selection Count")
plt.title("Roulette Wheel Selection")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
# plt.suptitle("Selection Methods Visualization", fontsize=16, y=1.02)
plt.show()

# Print fitness scores and selection results
print("Fitness Scores:", fitness_scores)
print("\nTournament Selection Counts:", tournament_selection_counts)
print("\nRoulette Wheel Selection Counts:", roulette_selection_counts)