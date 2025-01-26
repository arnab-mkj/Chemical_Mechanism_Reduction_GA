import matplotlib.pyplot as plt
import numpy as np

class RealTimePlotter:
    def __init__(self):
        
        self.generations = []
        self.best_fitness = []
        self.mean_fitness = []
        self.mole_fractions = []
        #self.worst_fitness = []
        
        #initializing the plot
        self.fig, self.ax = plt.subplots()
        self.best_line, = self.ax.plot([],[], label="Best Fitness", color="green")
        self.mean_line, = self.ax.plot([],[], label="Mean Fitness", color="blue")
        #self.worst_line, = self.ax.plot([],[], label="Worst Fitness", color="red")
        self.ax.set_xlabel("Generation", fontsize=12)
        self.ax.set_ylabel("Fitness Score", fontsize=12)
        self.ax.set_title("Real-Time Fitness Evaluation", fontsize=14)
        self.ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
        
    def update(self, generation, stats, selected_plots):
        # Append new data
        self.generations.append(generation)
        self.best_fitness.append(stats['best_fitness'])
        self.mean_fitness.append(stats['mean_fitness'])
        #self.worst_fitness.append(stats['worst_fitness'])
        
        def smooth(data, window_size=5):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

        # Update plot data
        self.best_line.set_data(self.generations, self.best_fitness)
        self.mean_line.set_data(self.generations, self.mean_fitness)
        #self.worst_line.set_data(self.generations, self.worst_fitness)

        # Adjust plot limits
        self.ax.set_xlim(0, max(self.generations) + 1)
        self.ax.set_ylim(0, max(self.mean_fitness)+1)
        #self.ax.set_ylim(0, max(self.best_fitness + self.mean_fitness + self.worst_fitness))
        
        # Check if mole fractions should be plotted
        if 'Mole Fractions' in selected_plots:
            self.plot_mole_fractions(stats['mole_fractions'])
            
        # Redraw the plot
        plt.pause(0.01)



    def plot_mole_fractions(self, mole_fractions):
        # Clear previous mole fraction plots
        self.ax.clear()
        self.ax.set_title("Mole Fractions Over Generations", fontsize=14)
        self.ax.set_xlabel("Species", fontsize=12)
        self.ax.set_ylabel("Mole Fraction", fontsize=12)

        # Plot each species mole fraction
        for species, fractions in mole_fractions.items():
            self.ax.plot(fractions['generation'], fractions['value'], label=species)

        self.ax.legend()
        plt.draw()


    def show(self):  
        plt.show()
        
    def save(self, filename="fitness_plot.png"):
        """
        Save the current plot to a file.
        """
        self.fig.savefig(filename, bbox_inches="tight")
        print(f"Plot saved as {filename}")