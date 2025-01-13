import matplotlib.pyplot as plt
import numpy as np

class RealTimePlotter:
    def __init__(self):
        
        self.generations = []
        self.best_fitness = []
        self.mean_fitness = []
        #self.worst_fitness = []
        
        #initializing the plot
        self.fig, self.ax = plt.subplots()
        self.best_line, = self.ax.plot([],[], label="Best Fitness", color="green")
        self.mean_line, = self.ax.plot([],[], label="Mean Fitness", color="blue")
        #self.worst_line, = self.ax.plot([],[], label="Worst Fitness", color="red")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness Score")
        self.ax.set_title("Real-Time Fitness Evaluation")
        self.ax.legend()
        self.ax.grid()
        
    def update(self, generation, stats):
        # Append new data
        self.generations.append(generation)
        self.best_fitness.append(stats['best_fitness'])
        self.mean_fitness.append(stats['mean_fitness'])
        #self.worst_fitness.append(stats['worst_fitness'])

        # Update plot data
        self.best_line.set_data(self.generations, self.best_fitness)
        self.mean_line.set_data(self.generations, self.mean_fitness)
        #self.worst_line.set_data(self.generations, self.worst_fitness)

        # Adjust plot limits
        self.ax.set_xlim(0, max(self.generations) + 1)
        self.ax.set_ylim(0, 1000)
        #self.ax.set_ylim(0, max(self.best_fitness + self.mean_fitness + self.worst_fitness))

        # Redraw the plot
        plt.pause(0.01)

    def show(self):
        """
        Display the final plot.
        """
        plt.show()