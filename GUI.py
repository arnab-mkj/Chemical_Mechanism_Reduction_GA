import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import importlib
import sys
from src.ChemicalMechanismGA.components.genetic_algorithm import GeneticAlgorithm
from src.ChemicalMechanismGA.components.fitness_function import evaluate_fitness
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
from src.ChemicalMechanismGA.utils.visualization import RealTimePlotter 
import json
import os
import numpy as np

class MechanismReductionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chemical Mechanism Reduction Dashboard")
        # Create input fields
        self.create_input_fields()
        
        self.create_reload_button()
        # Create run button
        self.run_button = tk.Button(root, text="Run Simulation", command=self.run_simulation)
        self.run_button.grid(row=10, column=1, pady=10)

        # Initialize species checkboxes
        self.species_checkboxes = {}
        self.create_species_checkboxes()

        # Initialize plot options checkboxes
        self.plot_options = {}
        self.create_plot_options()

    def create_input_fields(self):
        # Population Size
        tk.Label(self.root, text="Population Size:").grid(row=0, column=0)
        self.population_size_entry = tk.Entry(self.root)
        self.population_size_entry.grid(row=0, column=1)

        # Crossover Rate
        tk.Label(self.root, text="Crossover Rate (0-1):").grid(row=1, column=0)
        self.crossover_rate_entry = tk.Entry(self.root)
        self.crossover_rate_entry.grid(row=1, column=1)

        # Mutation Rate
        tk.Label(self.root, text="Mutation Rate (0-1):").grid(row=2, column=0)
        self.mutation_rate_entry = tk.Entry(self.root)
        self.mutation_rate_entry.grid(row=2, column=1)

        # Number of Generations
        tk.Label(self.root, text="Number of Generations:").grid(row=3, column=0)
        self.num_generations_entry = tk.Entry(self.root)
        self.num_generations_entry.grid(row=3, column=1)

        # Reactor Type
        tk.Label(self.root, text="Reactor Type:").grid(row=4, column=0)
        self.reactor_type_entry = ttk.Combobox(self.root, values=["constant_pressure", "batch", "1D-flame"])
        self.reactor_type_entry.grid(row=4, column=1)

        # Initial Temperature
        tk.Label(self.root, text="Initial Temperature (K):").grid(row=5, column=0)
        self.initial_temp_entry = tk.Entry(self.root)
        self.initial_temp_entry.grid(row=5, column=1)

        # Initial Pressure
        tk.Label(self.root, text="Initial Pressure (Pa):").grid(row=6, column=0)
        self.initial_pressure_entry = tk.Entry(self.root)
        self.initial_pressure_entry.grid(row=6, column=1)

        # Output Directory
        tk.Label(self.root, text="Output Directory:").grid(row=7, column=0)
        self.output_directory_entry = tk.Entry(self.root)
        self.output_directory_entry.grid(row=7, column=1)
        self.output_directory_entry.insert(0, "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/output")
        
    def create_reload_button(self):
        # Add a "Reload" button to the GUI
        reload_button = tk.Button(self.root, text="Reload GUI", command=self.reload_gui)
        reload_button.grid(row=11, column=1, pady=10)
        
    def reload_gui(self):
        # Restart the GUI using subprocess
        try:
            # Get the current Python executable and script path
            python = sys.executable
            script = os.path.abspath(sys.argv[0])

            # Start a new process
            subprocess.Popen([python, script])

            # Close the current GUI
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to restart GUI: {str(e)}")

    def create_species_checkboxes(self):
        # Example species list (this should be dynamically generated based on our mechanism)
        species_list = ["CO2", "H2O", "CH4", "O2", "N2"]
        tk.Label(self.root, text="Select Target Species:").grid(row=8, column=0, columnspan=2)

        for species in species_list:
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(self.root, text=species, variable=var)
            checkbox.grid(sticky='w')
            self.species_checkboxes[species] = var

    def create_plot_options(self):
        tk.Label(self.root, text="Select Plots to Display:").grid(row=9, column=0, columnspan=2)

        plot_options = ["Best Fitness", "Mean Fitness", "Mole Fractions"]
        for option in plot_options:
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(self.root, text=option, variable=var)
            checkbox.grid(sticky='w')
            self.plot_options[option] = var

    def run_simulation(self):
        try:
            # Gather inputs
            population_size = int(self.population_size_entry.get())
            crossover_rate = float(self.crossover_rate_entry.get())
            mutation_rate = float(self.mutation_rate_entry.get())
            num_generations = int(self.num_generations_entry.get())
            reactor_type = self.reactor_type_entry.get()
            initial_temp = float(self.initial_temp_entry.get())
            initial_pressure = float(self.initial_pressure_entry.get())
            output_directory = self.output_directory_entry.get()

            # Gather selected target species
            target_species = {species: 0.1 for species, var in self.species_checkboxes.items() if var.get()}
            
            selected_plots = [option for option, var in self.plot_options.items() if var.get()]

            # Initialize FitnessEvaluator
            fitness_evaluator = FitnessEvaluator(
                target_temperature=2000.0,  # Example target temperature
                target_species=target_species,  # User-defined target species
                target_delay=1.0,  # Example target ignition delay
                weight_temperature=1.0,  # Default weight for temperature fitness
                weight_species=1.0,  # Default weight for species fitness
                weight_ignition_delay=1.0,  # Default weight for ignition delay fitness
                difference_function="logarithmic",  # Default difference function
                sharpening_factor=10.0  # Empirical value
            )

            # Create GA instance
            ga = GeneticAlgorithm(
                population_size=population_size,
                genome_length=325,  # Assuming GRI-Mech 3.0 has 325 reactions
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                num_generations=num_generations,
                elite_size=2
            )
            
            plotter = RealTimePlotter()

            # Run the Genetic Algorithm
            for generation in range(num_generations):
                best_genome, best_fitness = ga.evolve(
                    fitness_function=lambda genome, 
                    original_mechanism_path="E:/PPP_WS2024-25/ChemicalMechanismReduction/data/gri30.yaml",
                    reactor_type=reactor_type, 
                    initial_temperature=initial_temp, 
                    initial_pressure=initial_pressure, 
                    generation=generation, 
                    filename=None: evaluate_fitness(
                        genome,
                        original_mechanism_path=original_mechanism_path,
                        reactor_type=reactor_type,
                        initial_temperature=initial_temp,  # Pass user-provided initial temperature
                        initial_pressure=initial_pressure,  # Pass user-provided initial pressure
                        generation=generation,
                        filename=filename
                    ),
                    original_mechanism_path="E:/PPP_WS2024-25/ChemicalMechanismReduction/data/gri30.yaml",
                    output_directory=output_directory,
                    reactor_type=reactor_type,
                    selected_plots=selected_plots
                )
                
                # Collect statistics for plotting
                stats = {
                    'best_fitness': best_fitness,
                    'mean_fitness': np.mean(ga.fitness_history),  # Assuming fitness_history is available
                    'mole_fractions': {
                        'generation': generation,
                        'value': best_genome  # Replace with actual mole fractions if available
                    }
                }
                
                
                plotter.update(generation, stats, selected_plots)

            # Save results
            output_path = os.path.join(output_directory, "best_reduced_mechanism.yaml")
            save_genome_as_yaml(best_genome, original_mechanism_path, output_path)

            # Save fitness history
            fitness_history_path = os.path.join(output_directory, "fitness_history.json")
            with open(fitness_history_path, "w") as file:
                json.dump(ga.fitness_history, file, indent=4)

            # Show success message
            messagebox.showinfo("Success", f"Best fitness: {best_fitness}\nResults saved to {output_path}")
            

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = MechanismReductionApp(root)
    root.mainloop()