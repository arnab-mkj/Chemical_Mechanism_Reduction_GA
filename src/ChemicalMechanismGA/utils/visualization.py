import matplotlib.pyplot as plt
import numpy as np
import os

class RealTimePlotter:
    def __init__(self):
        self.generations = []
        self.min_reactions = []  # To store the minimum number of reactions
        self.best_fitness = []
        self.mean_reactions = []
        self.mean_fitness = []
        self.total_fitness = []

        # New lists for fitness contributions
        self.temperature_fitness = []
        self.species_fitness = []
        self.idt_fitness = []
        self.reaction_fitness = []

        # Create figure for reactions
        self.fig1, self.ax1 = plt.subplots()
        self.reaction_line, = self.ax1.plot([], [], label="Min Reactions", color="orange")
        self.mean_reaction_line, = self.ax1.plot([], [], label="Mean Reactions", color="green", linestyle="--")
        self.ax1.set_xlabel("Generation", fontsize=6)
        self.ax1.set_ylabel("Number of Reactions", fontsize=6)
        self.ax1.set_title("Minimum Number of Reactions Over Generations", fontsize=14)
        self.ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        self.ax1.grid(color='gray', linestyle='--', linewidth=0.5)

        # Create figure for fitness
        self.fig2, self.ax2 = plt.subplots()
        self.fitness_line, = self.ax2.plot([], [], label="Best Fitness", color="blue")
        self.mean_fitness_line, = self.ax2.plot([], [], label="Mean Fitness", color="red")

        # New lines for fitness contributions
        self.temp_fitness_line, = self.ax2.plot([], [], label="Temperature Fitness", color="purple", linestyle="--", linewidth=0.9)
        self.species_fitness_line, = self.ax2.plot([], [], label="Species Fitness", color="green", linestyle="--", linewidth=0.9)
        self.idt_fitness_line, = self.ax2.plot([], [], label="IDT Fitness", color="orange", linestyle="--", linewidth=0.9)
        self.reaction_fitness_line, = self.ax2.plot([], [], label="Reaction Fitness", color="blue", linestyle="--", linewidth=0.9)
       

        self.ax2.set_xlabel("Generation", fontsize=12)
        self.ax2.set_ylabel("Fitness", fontsize=12)
        self.ax2.set_title("Best Fitness and Fitness Contributions Over Generations", fontsize=14)
        self.ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        self.ax2.grid(color='gray', linestyle='--', linewidth=0.5)

    def update(self, generation, stats):
        # Append new data
        self.generations.append(generation)
        self.min_reactions.append(stats["current_fitness_reactions"])  # Append the minimum number of reactions
        self.mean_reactions.append(stats["active_reactions_mean"])  # Append the mean number of reactions
        self.best_fitness.append(stats["total_best_fitness"])  # Append the best fitness value
        self.mean_fitness.append(stats["mean_fitness"]) # Append the mean fitness value
      

        # Append fitness contributions
        self.temperature_fitness.append(stats["temperature_fitness_min"])
        self.species_fitness.append(stats["species_fitness_min"])
        self.idt_fitness.append(stats["ignition_delay_fitness_min"])
        self.reaction_fitness.append(stats["reaction_fitness_min"])  # reaction fitness is related to min_reactions

        # Update reaction plot
        self.reaction_line.set_data(self.generations, self.min_reactions)
        self.mean_reaction_line.set_data(self.generations, self.mean_reactions)
        self.ax1.set_xlim(1, max(self.generations) + 1)
        self.ax1.set_ylim(0, max(self.min_reactions) * 1.1)  # Add padding to y-axis

        # Update fitness plot
        self.fitness_line.set_data(self.generations, self.best_fitness)
        self.mean_fitness_line.set_data(self.generations, self.mean_fitness)
        self.temp_fitness_line.set_data(self.generations, self.temperature_fitness)
        self.species_fitness_line.set_data(self.generations, self.species_fitness)
        self.idt_fitness_line.set_data(self.generations, self.idt_fitness)
        self.reaction_fitness_line.set_data(self.generations, self.reaction_fitness)
  

        self.ax2.set_xlim(1, max(self.generations) + 1)
        self.ax2.set_ylim(0, max(self.best_fitness) * 1.1)  # Add padding to y-axis

        # Redraw the plots
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()
        plt.pause(0.01)

    def show(self):
        """Show the plots."""
        plt.show()

    def save(self, fitness, crossover_rate, mutation_rate, diff_func):
        """
        Save the current plots to files with custom names.

        Parameters:
            fitness (float): Best fitness value.
            crossover_rate (float): Crossover success rate.
            mutation_rate (float): Mutation success rate.
            reactor_type (str): Reactor type (e.g., "constant_pressure").
        """
        output_dir = f"outputs/{diff_func}"
        os.makedirs(output_dir, exist_ok=True)
        # Generate custom filenames
        reaction_filename = f"{output_dir}/reaction_{diff_func}_c_{crossover_rate:.2f}_m_{mutation_rate:.2f}.png"
        fitness_filename = f"{output_dir}/fitness_{diff_func}_c_{crossover_rate:.2f}_m_{mutation_rate:.2f}.png"

        # Save the plots
        self.fig1.savefig(reaction_filename, bbox_inches="tight")
        self.fig2.savefig(fitness_filename, bbox_inches="tight")

        print(f"Reaction plot saved as {reaction_filename}")
        print(f"Fitness plot saved as {fitness_filename}")