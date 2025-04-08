import matplotlib.pyplot as plt
import numpy as np

class RealTimePlotter:
    def __init__(self):
        self.generations = []
        self.min_reactions = []  # To store the minimum number of reactions
        self.fig, self.ax = plt.subplots()

        # Reaction plot
        self.reaction_line, = self.ax.plot([], [], label="Min Reactions", color="orange")
        self.ax.set_xlabel("Generation", fontsize=12)
        self.ax.set_ylabel("Number of Reactions", fontsize=12)
        self.ax.set_title("Minimum Number of Reactions Over Generations", fontsize=14)
        self.ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        self.ax.grid(color='gray', linestyle='--', linewidth=0.5)

    def update(self, generation, stats):
        # Append new data
        self.generations.append(generation)
        self.min_reactions.append(stats["min_reactions"])  # Append the minimum number of reactions

        # Update reaction plot
        self.reaction_line.set_data(self.generations, self.min_reactions)
        self.ax.set_xlim(1, max(self.generations) + 1)
        self.ax.set_ylim(0, 325)

        # Redraw the plot
        plt.pause(0.01)

    def show(self):
        """Show the plot."""
        plt.show()

    def save(self, reaction_filename="reaction_plot.png"):
        """
        Save the current plot to a file.
        """
        self.fig.savefig(reaction_filename, bbox_inches="tight")
        print(f"Reaction plot saved as {reaction_filename}")