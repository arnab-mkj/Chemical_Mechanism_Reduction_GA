import numpy as np
import matplotlib.pyplot as plt
import math


def penalized_fitness(reaction_count, sigma, lam, max_reactions=325):
  
    return 1 / (1 + math.exp(sigma * (1 - (reaction_count / (lam * max_reactions)))))

# Define ranges for testing
reaction_counts = np.linspace(0, 325, 325)  # Reaction count range
sigma_values = [1, 2, 4, 6, 8]  # Sharpening factors
lambda_values = [0.3, 0.5, 0.8, 1.0]  # Shift parameters

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot penalized fitness for different sigma and lambda values
for sigma in sigma_values:
    for lam in lambda_values:
        fitness_values = [penalized_fitness(rc, sigma, lam) for rc in reaction_counts]
        ax.plot(
            reaction_counts,
            fitness_values,
            label=f"σ = {sigma}, λ = {lam}",
            alpha=0.8
        )

# Customize the plot
ax.set_title("Penalized Fitness vs. Reaction Count", fontsize=14)
ax.set_xlabel("Reaction Count", fontsize=12)
ax.set_ylabel("Penalized Fitness", fontsize=12)
ax.grid(color="gray", linestyle="--", linewidth=0.5)
ax.legend(title="Parameters", fontsize=10, loc="best")

# Show the plot
plt.tight_layout()
plt.show()