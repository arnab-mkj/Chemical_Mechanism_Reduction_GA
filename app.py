import json
import cantera as ct
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from src.ChemicalMechanismGA.components.genetic_algorithm import GeneticAlgorithm
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator
from src.ChemicalMechanismGA.utils.save_best_genome import save_genome_as_yaml
import time
import os
import numpy as np
import importlib.util
from src.ChemicalMechanismGA.utils.species_sens_weights import compute_species_weights
from src.ChemicalMechanismGA.utils.sensitivity_reduce import MechanismReducer
from extras.premix_reactor import PremixFlameComparison 
from src.ChemicalMechanismGA.tests.test_full_red_compare import MechanismComparison

# Main GUI class for the genetic algorithm application
class GeneticAlgorithmGUI:
    def __init__(self, root):
        # Store the root Tkinter window and set the title and size of the GUI window
        self.root = root
        self.root.title("Genetic Algorithm for Chemical Mechanism Reduction")
        self.root.geometry("1000x800")
        
        # Create a canvas for scrollable content and a vertical scrollbar
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        # Frame inside the canvas to hold all widgets
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Bind the scrollable frame to update the canvas scroll region when its size changes
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        # Place the frame inside the canvas at the top-left corner
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        # Link the canvas to the scrollbar for vertical scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack the scrollbar on the right side and the canvas on the left, filling available space
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a notebook (tabbed interface) to organize tabs
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create the Configuration tab frame with padding for better spacing
        self.main_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_frame, text="Configuration")
        
        # Create the Tests tab frame with padding
        self.tests_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tests_frame, text="Tests")
        
        # Create the Flame Simulation tab frame with padding
        self.flame_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.flame_frame, text="Flame Simulation")
        
        # Create the Constant Pressure Simulation tab frame with padding
        self.const_pressure_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.const_pressure_frame, text="Constant Pressure Simulation")
        
        # Dictionary to store configuration variables as Tkinter StringVar or BooleanVar objects
        self.config_vars = {
            "population_size": tk.StringVar(value="50"),
            "genome_length": tk.StringVar(value="325"),
            "crossover_rate": tk.StringVar(value="0.6"),
            "mutation_rate": tk.StringVar(value="0.005"),
            "num_generations": tk.StringVar(value="100"),
            "elite_size": tk.StringVar(value="2"),
            "original_mechanism_path": tk.StringVar(value="gri30.yaml"),
            "output_directory": tk.StringVar(value="./output_gui"),
            "reactor_type": tk.StringVar(value="constant_pressure"),
            "difference_function": tk.StringVar(value="squared"),
            "sharpening_factor": tk.StringVar(value="10.0"),
            "normalization_method": tk.StringVar(value="sigmoid"),
            "temperature_weight": tk.StringVar(value="1.0"),
            "IDT_weight": tk.StringVar(value="1.0"),
            "reactions_weight": tk.StringVar(value="1.0"),
            "conditions": tk.StringVar(value='{"temperature": 1800, "pressure": 1e5, "equivalence_ratio": 1.0, "fuel": "CH4", "oxidizer": "O2:0.21, N2:0.79"}'),
            "key_species": tk.StringVar(value='["CH4", "O2"]'),
            "species_weights_input": tk.StringVar(value='{"CH4": 0.5, "O2": 0.5}'),
            "weights_method": tk.StringVar(value="manual"),
            "elitism_enabled": tk.BooleanVar(value=True),
            "deactivation_chance": tk.StringVar(value="0.0"),
            "init_with_reduced_mech": tk.BooleanVar(value=False),
            "reduction_threshold": tk.StringVar(value="0.1")
        }
        
        # Dictionary to store test-related variables
        self.test_vars = {
            "full_mech": tk.StringVar(value="gri30.yaml"),
            "reduced_mech": tk.StringVar(value="reduced_random_gri30.yaml"),
            "reduction_fraction": tk.StringVar(value="0.3"),
            "mutation_rate": tk.StringVar(value="0.01"),
            "temperature": tk.StringVar(value="1800"),
            "pressure": tk.StringVar(value="1e5"),
            "equivalence_ratio": tk.StringVar(value="1.0"),
            "fuel": tk.StringVar(value="CH4"),
            "oxidizer": tk.StringVar(value="O2:0.21, N2:0.79"),
            "diversity_prob": tk.StringVar(value="0.05"),
            "remove_reactions": tk.StringVar(value='["H + O2 <=> O + OH", "HO2 + OH <=> H2O + O2"]'),
            "crossover_rate": tk.StringVar(value="0.6"),
            "genome_length": tk.StringVar(value="20"),
            "population_size": tk.StringVar(value="10")
        }
        
        # Dictionary to store flame simulation variables
        self.flame_vars = {
            "reactor_type": tk.StringVar(value="Laminar Flame Reactor"),
            "phi": tk.StringVar(value="2.0"),
            "temperature": tk.StringVar(value="1000"),
            "pressure": tk.StringVar(value=str(ct.one_atm)),
            "fuel": tk.StringVar(value="CH4"),
            "oxidizer": tk.StringVar(value="O2:0.21, N2:0.79"),
            "mass_flow_rate": tk.StringVar(value="0.04"),
            "reduced_mech_path": tk.StringVar(value=""),
            "full_mech_name": tk.StringVar(value="gri30.yaml")
        }
        
        # Dictionary to store constant pressure simulation variables
        self.const_pressure_vars = {
            "temperature": tk.StringVar(value="2561"),
            "pressure": tk.StringVar(value=str(ct.one_atm)),
            "equivalence_ratios": tk.StringVar(value="[0.4]"),
            "fuel": tk.StringVar(value="CH4"),
            "oxidizer": tk.StringVar(value="O2:0.21, N2:0.79"),
            "end_time": tk.StringVar(value="0.1"),
            "key_species": tk.StringVar(value='["CH4", "O2", "CO2", "CO", "OH"]'),
            "reduced_mech_path": tk.StringVar(value=""),
            "full_mech_name": tk.StringVar(value="gri30.yaml"),
            "output_dir": tk.StringVar(value="./output_gui"),
            "xlim": tk.StringVar(value="0.1")
        }
        
        # Initialize the GUI input fields and interfaces
        self.create_input_fields()
        self.create_test_interface()
        self.create_flame_interface()
        self.create_const_pressure_interface()
        
        # Create a text widget to display output messages in the Configuration tab
        self.output_text = tk.Text(self.main_frame, height=10, width=80)
        self.output_text.grid(row=len(self.config_vars) + 6, column=0, columnspan=3, pady=10)
        
        # Add a button to run the genetic algorithm
        ttk.Button(self.main_frame, text="Run Genetic Algorithm", command=self.run_ga).grid(row=len(self.config_vars) + 7, column=0, columnspan=3, pady=10)
    
    def create_input_fields(self):
        row = 0
        ttk.Label(self.main_frame, text="Species Weights Method").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(self.main_frame, text="Manual Input", variable=self.config_vars["weights_method"], value="manual", command=self.toggle_weights_input).grid(row=row, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(self.main_frame, text="Sensitivity Analysis", variable=self.config_vars["weights_method"], value="sensitivity", command=self.toggle_weights_input).grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1
        
        self.species_weights_label = ttk.Label(self.main_frame, text="Species Weights (JSON)")
        self.species_weights_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.species_weights_text = tk.Text(self.main_frame, height=4, width=50)
        self.species_weights_text.insert(tk.END, self.config_vars["species_weights_input"].get())
        self.species_weights_text.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        self.species_weights_text.bind("<KeyRelease>", lambda e: self.config_vars["species_weights_input"].set(self.species_weights_text.get("1.0", tk.END).strip()))
        row += 1
        
        ttk.Checkbutton(self.main_frame, text="Enable Elitism", variable=self.config_vars["elitism_enabled"], command=self.toggle_elite_size).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        self.elite_size_label = ttk.Label(self.main_frame, text="Elite Size")
        self.elite_size_label.grid(row=row + 1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.main_frame, textvariable=self.config_vars["elite_size"], width=50, state="disabled").grid(row=row + 1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 2
        
        ttk.Label(self.main_frame, text="Deactivation Chance (%) for Full Mechanism").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.main_frame, textvariable=self.config_vars["deactivation_chance"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        ttk.Checkbutton(self.main_frame, text="Initialize with Reduced Mechanism from Species Sensitivity", variable=self.config_vars["init_with_reduced_mech"], command=self.toggle_reduction_threshold).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1
        self.reduction_threshold_label = ttk.Label(self.main_frame, text="Reduction Threshold (fraction of least active reactions to remove, e.g., 0.1 for 10%)")
        self.reduction_threshold_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.main_frame, textvariable=self.config_vars["reduction_threshold"], width=50, state="disabled").grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Add OptionMenu for Reactor Type
        ttk.Label(self.main_frame, text="Reactor Type").grid(row=row, column=0, sticky=tk.W, pady=2)
        reactor_options = ["constant_pressure"]
        self.reactor_menu = ttk.OptionMenu(self.main_frame, self.config_vars["reactor_type"], self.config_vars["reactor_type"].get(), *reactor_options)
        self.reactor_menu.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Add OptionMenu for Difference Function
        ttk.Label(self.main_frame, text="Difference Function").grid(row=row, column=0, sticky=tk.W, pady=2)
        diff_options = ["absolute", "squared"]
        self.diff_menu = ttk.OptionMenu(self.main_frame, self.config_vars["difference_function"], self.config_vars["difference_function"].get(), *diff_options)
        self.diff_menu.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Add OptionMenu for Normalization Method
        ttk.Label(self.main_frame, text="Normalization Method").grid(row=row, column=0, sticky=tk.W, pady=2)
        norm_options = ["sigmoid", "logarithmic"]
        self.norm_menu = ttk.OptionMenu(self.main_frame, self.config_vars["normalization_method"], self.config_vars["normalization_method"].get(), *norm_options)
        self.norm_menu.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        for key, var in self.config_vars.items():
            if key in ["species_weights_input", "weights_method", "elitism_enabled", "deactivation_chance", "init_with_reduced_mech", "elite_size", "reduction_threshold", "reactor_type", "difference_function", "normalization_method"]:
                continue
            label_text = key.replace("_", " ").title()
            ttk.Label(self.main_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
            
            if key == "output_directory":
                ttk.Entry(self.main_frame, textvariable=var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
                ttk.Button(self.main_frame, text="Browse", command=lambda k=key: self.browse_file(k)).grid(row=row, column=2, pady=2)
            elif key == "conditions":
                text_area = tk.Text(self.main_frame, height=4, width=50)
                text_area.insert(tk.END, var.get())
                text_area.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
                text_area.bind("<KeyRelease>", lambda e, v=var, t=text_area: v.set(t.get("1.0", tk.END).strip()))
            else:
                ttk.Entry(self.main_frame, textvariable=var, width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            
            row += 1
    
    def create_test_interface(self):
        row = 0
        ttk.Label(self.tests_frame, text="Select Test to Run").grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        self.test_var = tk.StringVar(value="mutation")
        tests = ["mutation", "fitness", "initialization", "crossover", "selection"]
        self.test_menu = ttk.OptionMenu(self.tests_frame, self.test_var, "mutation", *tests, command=self.update_test_fields)
        self.test_menu.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.tests_frame, text="Run Test", command=self.run_test).grid(row=row, column=2, pady=2)
        row += 1
        
        self.test_fields_frame = ttk.Frame(self.tests_frame)
        self.test_fields_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        self.output_text = tk.Text(self.tests_frame, height=10, width=80)
        self.output_text.grid(row=row, column=0, columnspan=3, pady=10)
        
        self.update_test_fields("mutation")
    
    def create_flame_interface(self):
        row = 0
        ttk.Label(self.flame_frame, text="Select Reactor Type").grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        reactor_types = ["Laminar Flame Reactor"]
        self.reactor_var = tk.StringVar(value="Laminar Flame Reactor")
        ttk.OptionMenu(self.flame_frame, self.reactor_var, "Laminar Flame Reactor", *reactor_types, command=self.update_flame_fields).grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        self.flame_fields_frame = ttk.Frame(self.flame_frame)
        self.flame_fields_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        self.flame_output_text = tk.Text(self.flame_frame, height=10, width=80)
        self.flame_output_text.grid(row=row, column=0, columnspan=3, pady=10)
        
        ttk.Button(self.flame_frame, text="Run Flame Simulation", command=self.run_flame_simulation).grid(row=row + 1, column=0, columnspan=3, pady=10)
        
        self.update_flame_fields("Laminar Flame Reactor")
    
    def create_const_pressure_interface(self):
        row = 0
        ttk.Label(self.const_pressure_frame, text="Constant Pressure Reactor Simulation").grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        self.const_pressure_fields_frame = ttk.Frame(self.const_pressure_frame)
        self.const_pressure_fields_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Create input fields for constant pressure simulation
        ttk.Label(self.const_pressure_fields_frame, text="Temperature (K)").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["temperature"], width=50).grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="Pressure (Pa)").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["pressure"], width=50).grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="Equivalence Ratios (JSON list, e.g., [0.4, 0.8])").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["equivalence_ratios"], width=50).grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="Fuel").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["fuel"], width=50).grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="Oxidizer (e.g., O2:0.21, N2:0.79)").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["oxidizer"], width=50).grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="End Time (s)").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["end_time"], width=50).grid(row=5, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="Key Species (JSON list, e.g., ['CH4', 'O2'])").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["key_species"], width=50).grid(row=6, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="Reduced Mechanism File").grid(row=7, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["reduced_mech_path"], width=40).grid(row=7, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.const_pressure_fields_frame, text="Browse", command=lambda: self.browse_file("reduced_mech_path", self.const_pressure_vars)).grid(row=7, column=2, pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="Full Mechanism Name (e.g., gri30.yaml)").grid(row=8, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["full_mech_name"], width=50).grid(row=8, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="Output Directory").grid(row=9, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["output_dir"], width=40).grid(row=9, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.const_pressure_fields_frame, text="Browse", command=lambda: self.browse_file("output_dir", self.const_pressure_vars, is_dir=True)).grid(row=9, column=2, pady=2)
        
        ttk.Label(self.const_pressure_fields_frame, text="X-Axis Limit (ms)").grid(row=10, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.const_pressure_fields_frame, textvariable=self.const_pressure_vars["xlim"], width=50).grid(row=10, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        self.const_pressure_output_text = tk.Text(self.const_pressure_frame, height=10, width=80)
        self.const_pressure_output_text.grid(row=row, column=0, columnspan=3, pady=10)
        
        ttk.Button(self.const_pressure_frame, text="Run Constant Pressure Simulation", command=self.run_const_pressure_simulation).grid(row=row + 1, column=0, columnspan=3, pady=10)
    
    def update_test_fields(self, test_type):
        for widget in self.test_fields_frame.winfo_children():
            widget.destroy()
        
        row = 0
        if test_type == "mutation":
            ttk.Label(self.test_fields_frame, text="Mutation Rate").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["mutation_rate"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
        
        elif test_type == "fitness":
            ttk.Label(self.test_fields_frame, text="Full Mechanism File").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["full_mech"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Reduced Mechanism File").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["reduced_mech"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Reduction Fraction").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["reduction_fraction"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Temperature (K)").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["temperature"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Pressure (Pa)").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["pressure"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Equivalence Ratio").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["equivalence_ratio"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Fuel").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["fuel"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Oxidizer").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["oxidizer"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
        
        elif test_type == "initialization":
            ttk.Label(self.test_fields_frame, text="Diversity Probability").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["diversity_prob"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Remove Reactions (JSON list)").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["remove_reactions"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
        
        elif test_type == "crossover":
            ttk.Label(self.test_fields_frame, text="Crossover Rate").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["crossover_rate"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Genome Length").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["genome_length"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
            ttk.Label(self.test_fields_frame, text="Population Size").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.test_fields_frame, textvariable=self.test_vars["population_size"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
        
        elif test_type == "selection":
            pass
    
    def update_flame_fields(self, reactor_type):
        for widget in self.flame_fields_frame.winfo_children():
            widget.destroy()
        
        row = 0
        ttk.Label(self.flame_fields_frame, text="Equivalence Ratio (phi)").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.flame_fields_frame, textvariable=self.flame_vars["phi"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        ttk.Label(self.flame_fields_frame, text="Temperature (K)").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.flame_fields_frame, textvariable=self.flame_vars["temperature"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        ttk.Label(self.flame_fields_frame, text="Pressure (Pa)").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.flame_fields_frame, textvariable=self.flame_vars["pressure"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        ttk.Label(self.flame_fields_frame, text="Fuel").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.flame_fields_frame, textvariable=self.flame_vars["fuel"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        ttk.Label(self.flame_fields_frame, text="Oxidizer (e.g., O2:0.21, N2:0.79)").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.flame_fields_frame, textvariable=self.flame_vars["oxidizer"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        if reactor_type == "Laminar Flame Reactor":
            ttk.Label(self.flame_fields_frame, text="Mass Flow Rate (kg/m²/s)").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(self.flame_fields_frame, textvariable=self.flame_vars["mass_flow_rate"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
            row += 1
        
        ttk.Label(self.flame_fields_frame, text="Reduced Mechanism File").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.flame_fields_frame, textvariable=self.flame_vars["reduced_mech_path"], width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(self.flame_fields_frame, text="Browse", command=lambda: self.browse_file("reduced_mech_path", self.flame_vars)).grid(row=row, column=2, pady=2)
        row += 1
        
        ttk.Label(self.flame_fields_frame, text="Full Mechanism Name (e.g., gri30.yaml)").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.flame_fields_frame, textvariable=self.flame_vars["full_mech_name"], width=50).grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        row += 1
    
    def toggle_elite_size(self):
        state = "normal" if self.config_vars["elitism_enabled"].get() else "disabled"
        self.elite_size_label.configure(state=state)
        ttk.Entry(self.main_frame, textvariable=self.config_vars["elite_size"], width=50, state=state).grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
    
    def toggle_reduction_threshold(self):
        state = "normal" if self.config_vars["init_with_reduced_mech"].get() else "disabled"
        self.reduction_threshold_label.configure(state=state)
        ttk.Entry(self.main_frame, textvariable=self.config_vars["reduction_threshold"], width=50, state=state).grid(row=5, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        if self.config_vars["init_with_reduced_mech"].get():
            self.config_vars["deactivation_chance"].set("0.0")
    
    def toggle_weights_input(self):
        method = self.config_vars["weights_method"].get()
        if method == "manual":
            self.species_weights_label.grid()
            self.species_weights_text.grid()
        else:
            self.species_weights_label.grid_remove()
            self.species_weights_text.grid_remove()
    
    def browse_file(self, key, var_dict=None, is_dir=False):
        if var_dict is None:
            var_dict = self.config_vars
        if is_dir:
            path = filedialog.askdirectory(title=f"Select {key.replace('_', ' ').title()}")
        else:
            path = filedialog.askopenfilename(title=f"Select {key.replace('_', ' ').title()}")
        if path:
            var_dict[key].set(path)
    
    def normalize_weights(self, weights):
        try:
            total = sum(float(v) for v in weights.values())
            if total == 0:
                raise ValueError("Sum of weights cannot be zero")
            return {k: float(v) / total for k, v in weights.items()}
        except Exception as e:
            raise ValueError(f"Invalid weights: {str(e)}")
    
    def validate_inputs(self):
        try:
            for key in ["population_size", "genome_length", "elite_size", "num_generations"]:
                int(self.config_vars[key].get())
            for key in ["crossover_rate", "mutation_rate", "sharpening_factor", "temperature_weight", "IDT_weight", "reactions_weight", "deactivation_chance", "reduction_threshold"]:
                value = float(self.config_vars[key].get())
                if key == "deactivation_chance" and (value < 0 or value > 100):
                    raise ValueError("Deactivation chance must be between 0 and 100")
                if key == "reduction_threshold" and (value < 0 or value > 1):
                    raise ValueError("Reduction threshold must be between 0 and 1")
            
            mechanism_name = self.config_vars["original_mechanism_path"].get().strip()
            if not mechanism_name:
                raise ValueError("Mechanism name must be specified")
            
            output_dir = self.config_vars["output_directory"].get()
            if not output_dir:
                raise ValueError("Output directory must be specified")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            key_species = json.loads(self.config_vars["key_species"].get())
            if not key_species:
                raise ValueError("Key species list cannot be empty")
            
            conditions = json.loads(self.config_vars["conditions"].get())
            required_condition_keys = ["temperature", "pressure", "equivalence_ratio", "fuel", "oxidizer"]
            if not all(k in conditions for k in required_condition_keys):
                raise ValueError(f"Conditions must include: {', '.join(required_condition_keys)}")
            
            if self.config_vars["weights_method"].get() == "manual":
                weights = json.loads(self.config_vars["species_weights_input"].get())
                if not isinstance(weights, dict):
                    raise ValueError("Species weights must be a JSON object")
                if not all(float(v) >= 0 for v in weights.values()):
                    raise ValueError("Species weights must be non-negative")
                if not all(sp in weights for sp in key_species):
                    raise ValueError("All key species must have weights specified")
            
            try:
                ct.Solution(mechanism_name)
            except Exception as e:
                raise ValueError(f"Invalid mechanism name: {str(e)}")
            
            return True
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
            return False
    
    def save_config(self):
        try:
            conditions_str = self.config_vars["conditions"].get().strip()
            if not conditions_str:
                raise ValueError("Conditions field is empty")
            
            conditions = json.loads(conditions_str)
            required_condition_keys = ["temperature", "pressure", "equivalence_ratio", "fuel", "oxidizer"]
            if not all(k in conditions for k in required_condition_keys):
                raise ValueError(f"Conditions dictionary must contain: {', '.join(required_condition_keys)}")
            
            oxidizer_str = conditions["oxidizer"]
            oxidizer_dict = {}
            for pair in oxidizer_str.split(","):
                species, ratio = pair.strip().split(":")
                oxidizer_dict[species] = float(ratio)
            
            fuel_dict = {conditions["fuel"]: 1.0}
            
            transformed_conditions = {
                "phi": float(conditions["equivalence_ratio"]),
                "fuel": fuel_dict,
                "oxidizer": oxidizer_dict,
                "pressure": float(conditions["pressure"]),
                "temperature": float(conditions["temperature"])
            }
            
            conditions_list = [transformed_conditions]
            
            output_dir = self.config_vars["output_directory"].get()
            species_weights_path = os.path.join(output_dir, "species_weights.json")
            
            if self.config_vars["weights_method"].get() == "manual":
                species_weights = json.loads(self.config_vars["species_weights_input"].get())
                species_weights = self.normalize_weights(species_weights)
            else:
                key_species = json.loads(self.config_vars["key_species"].get())
                species_weights = compute_species_weights(
                    self.config_vars["original_mechanism_path"].get(),
                    conditions_list[0],
                    key_species,
                    species_weights_path
                )
            
            config = {
                "population_size": int(self.config_vars["population_size"].get()),
                "genome_length": int(self.config_vars["genome_length"].get()),
                "crossover_rate": float(self.config_vars["crossover_rate"].get()),
                "mutation_rate": float(self.config_vars["mutation_rate"].get()),
                "num_generations": int(self.config_vars["num_generations"].get()),
                "elite_size": int(self.config_vars["elite_size"].get()) if self.config_vars["elitism_enabled"].get() else 0,
                "original_mechanism_path": self.config_vars["original_mechanism_path"].get(),
                "output_directory": output_dir,
                "reactor_type": self.config_vars["reactor_type"].get(),
                "difference_function": self.config_vars["difference_function"].get(),
                "sharpening_factor": float(self.config_vars["sharpening_factor"].get()),
                "normalization_method": self.config_vars["normalization_method"].get(),
                "conditions": conditions_list,
                "key_species": json.loads(self.config_vars["key_species"].get()),
                "weights": {
                    "temperature": float(self.config_vars["temperature_weight"].get()),
                    "IDT": float(self.config_vars["IDT_weight"].get()),
                    "reactions": float(self.config_vars["reactions_weight"].get()),
                    "species": "species_weights.json"
                },
                "elitism_enabled": self.config_vars["elitism_enabled"].get(),
                "deactivation_chance": float(self.config_vars["deactivation_chance"].get()) / 100.0,
                "init_with_reduced_mech": self.config_vars["init_with_reduced_mech"].get(),
                "reduction_threshold": float(self.config_vars["reduction_threshold"].get())
            }
            
            if self.config_vars["weights_method"].get() == "manual":
                with open(species_weights_path, "w") as f:
                    json.dump(species_weights, f, indent=4)
            
            config_path = os.path.join(output_dir, "params.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            
            return config_path
        except json.JSONDecodeError as e:
            messagebox.showerror("Config Save Error", f"Invalid JSON in conditions or weights: {str(e)}")
            return None
        except Exception as e:
            messagebox.showerror("Config Save Error", f"Failed to save configuration: {str(e)}")
            return None
    
    def run_ga(self):
        if not self.validate_inputs():
            return
        
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Saving configuration and running Genetic Algorithm...\n")
        self.root.update()
        
        try:
            config_path = self.save_config()
            if not config_path:
                return
            
            start_time = time.time()
            
            with open(config_path, "r") as f:
                config = json.load(f)
            
            weights_file = os.path.join(config["output_directory"], config["weights"]["species"])
            with open(weights_file, "r") as f:
                species_weights = json.load(f)
            
            weights = {
                "temperature": config["weights"]["temperature"],
                "IDT": config["weights"]["IDT"],
                "species": species_weights,
                "reactions": config["weights"]["reactions"]
            }
            
            mechanism_path = config["original_mechanism_path"]
           
            if config["init_with_reduced_mech"]:
                self.output_text.insert(tk.END, "Creating reduced mechanism based on species sensitivity...\n")
                self.root.update()
                reducer = MechanismReducer(config_path)
                reduced_mech_file = reducer.create_reduced_mechanism(threshold=config["reduction_threshold"])
                mechanism_path = str(reduced_mech_file)
                self.output_text.insert(tk.END, f"Using reduced mechanism: {mechanism_path}\n")
                self.root.update()
                config["deactivation_chance"] = 0.0
            else:
                self.output_text.insert(tk.END, f"Using full mechanism with deactivation chance: {config['deactivation_chance']*100}%\n")
                self.root.update()
            
            fitness_evaluator = FitnessEvaluator(
                mechanism_path,
                config["reactor_type"],
                config["conditions"],
                weights,
                config["genome_length"],
                config["difference_function"],
                config["sharpening_factor"],
                config["normalization_method"],
                config["key_species"]
            )
            
            ga = GeneticAlgorithm(
                config["population_size"],
                config["genome_length"],
                config["crossover_rate"],
                config["mutation_rate"],
                config["num_generations"],
                config["elite_size"],
                fitness_evaluator,
                config["key_species"],
                config["difference_function"],
                elitism_enabled=config["elitism_enabled"],
                deactivation_chance=config["deactivation_chance"],
                init_with_reduced_mech=config["init_with_reduced_mech"]
            )
            
            best_genome, best_fitness = ga.evolve(config["output_directory"])
            
            output = f"\nBest solution found:\n"
            output += f"Fitness: {best_fitness}\n"
            output += f"Number of active reactions: {sum(best_genome)}\n"
            
            output_path = f"{config['output_directory']}/{config['difference_function']}"
            save_genome_as_yaml(best_genome, mechanism_path, output_path)
            output += f"Best reduced mechanism saved to: {output_path}\n"
            
            # fitness_history_converted = {
            #     gen: {k: float(v) if isinstance(v, np.integer) else v for k, v in values.items()}
            #     for gen, values in ga.fitness_history.items()
            # }
            fitness_history_path = f"{config['output_directory']}/{config['difference_function']}/fitness_history.json"
            with open(fitness_history_path, "w") as file:
                json.dump(ga.fitness_history, file, indent=4)
            output += f"Fitness history saved to: {fitness_history_path}\n"
            
            end_time = time.time()
            total_runtime = end_time - start_time
            output += f"\nTotal runtime: {total_runtime:.2f} seconds\n"
            
            self.output_text.insert(tk.END, output)
        except Exception as e:
            self.output_text.insert(tk.END, f"Error: {str(e)}\n")
            messagebox.showerror("Execution Error", f"An error occurred: {str(e)}")
    
    def run_test(self):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Running test: {self.test_var.get()}...\n")
        self.root.update()
        
        try:
            test_module = self.test_var.get()
            spec = importlib.util.spec_from_file_location(
                test_module,
                f"src/ChemicalMechanismGA/tests/test_{test_module}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if test_module == "mutation":
                mutation_rate = float(self.test_vars["mutation_rate"].get())
                result = module.run_test(mutation_rate=mutation_rate)
            
            elif test_module == "fitness":
                full_mech = self.test_vars["full_mech"].get()
                reduced_mech = self.test_vars["reduced_mech"].get()
                reduction_fraction = float(self.test_vars["reduction_fraction"].get())
                temperature = float(self.test_vars["temperature"].get())
                pressure = float(self.test_vars["pressure"].get())
                equivalence_ratio = float(self.test_vars["equivalence_ratio"].get())
                fuel_str = self.test_vars["fuel"].get()
                oxidizer_str = self.test_vars["oxidizer"].get()
                
                oxidizer_dict = {}
                for pair in oxidizer_str.split(","):
                    species, ratio = pair.strip().split(":")
                    oxidizer_dict[species] = float(ratio)
                
                key_species = json.loads(self.config_vars["key_species"].get())
                
                test_config = {
                    "full_mech": full_mech,
                    "reduced_mech": reduced_mech,
                    "reduction_fraction": reduction_fraction,
                    "temperature": temperature,
                    "pressure": pressure,
                    "equivalence_ratio": equivalence_ratio,
                    "fuel": {fuel_str: 1.0},
                    "oxidizer": oxidizer_dict,
                    "key_species": key_species
                }
                
                with open("params_test.json", "w") as f:
                    json.dump(test_config, f, indent=4)
                
                result = module.run_test(params_file="params_test.json")
            
            elif test_module == "initialization":
                diversity_prob = float(self.test_vars["diversity_prob"].get())
                remove_reactions = json.loads(self.test_vars["remove_reactions"].get())
                result = module.run_test(
                    diversity_prob=diversity_prob,
                    remove_reactions=remove_reactions
                )
            
            elif test_module == "crossover":
                crossover_rate = float(self.test_vars["crossover_rate"].get())
                genome_length = int(self.test_vars["genome_length"].get())
                population_size = int(self.test_vars["population_size"].get())
                if not (0 <= crossover_rate <= 1):
                    raise ValueError("Crossover rate must be between 0 and 1")
                if genome_length < 2:
                    raise ValueError("Genome length must be at least 2")
                if population_size < 2:
                    raise ValueError("Population size must be at least 2")
                result = module.run_test(
                    crossover_rate=crossover_rate,
                    genome_length=genome_length,
                    population_size=population_size
                )
                self.output_text.insert(tk.END, result['summary'])
            
            elif test_module == "selection":
                result = module.run_test()
            
            self.output_text.insert(tk.END, f"Test {self.test_var.get()} completed.\n")
        except Exception as e:
            self.output_text.insert(tk.END, f"Error running test {self.test_var.get()}: {str(e)}\n")
            messagebox.showerror("Test Error", f"An error occurred: {str(e)}")
    
    def run_flame_simulation(self):
        self.flame_output_text.delete(1.0, tk.END)
        self.flame_output_text.insert(tk.END, "Running flame simulation...\n")
        self.root.update()
        
        try:
            # Validate inputs
            phi = float(self.flame_vars["phi"].get())
            temperature = float(self.flame_vars["temperature"].get())
            pressure = float(self.flame_vars["pressure"].get())
            fuel = self.flame_vars["fuel"].get()
            oxidizer_str = self.flame_vars["oxidizer"].get()
            mass_flow_rate = float(self.flame_vars["mass_flow_rate"].get()) if self.reactor_var.get() == "Laminar Flame Reactor" else None
            reduced_mech_path = self.flame_vars["reduced_mech_path"].get()
            full_mech_name = self.flame_vars["full_mech_name"].get()
            
            if not reduced_mech_path or not os.path.exists(reduced_mech_path):
                raise ValueError("Invalid or missing reduced mechanism file")
            if not full_mech_name:
                raise ValueError("Invalid full mechanism name")
            try:
                ct.Solution(full_mech_name)
            except Exception as e:
                raise ValueError(f"Invalid full mechanism name: {str(e)}")
            if phi <= 0:
                raise ValueError("Equivalence ratio must be positive")
            if temperature <= 0:
                raise ValueError("Temperature must be positive")
            if pressure <= 0:
                raise ValueError("Pressure must be positive")
            if mass_flow_rate is not None and mass_flow_rate <= 0:
                raise ValueError("Mass flow rate must be positive")
            
            # Parse oxidizer
            oxidizer_dict = {}
            for pair in oxidizer_str.split(","):
                species, ratio = pair.strip().split(":")
                oxidizer_dict[species] = float(ratio)
            
            # Create condition dictionary
            condition = {
                "phi": phi,
                "temperature": temperature,
                "pressure": pressure,
                "fuel": fuel,
                "oxidizer": oxidizer_dict
            }
            if mass_flow_rate is not None:
                condition["mass_flow_rate"] = mass_flow_rate
            
            # Initialize gas with full mechanism
            gas = ct.Solution(full_mech_name)
            comparison = PremixFlameComparison(gas, reduced_mech_path, full_mech_name)
            
            # Determine reactor type and run simulation
            reactor_type = self.reactor_var.get()
            if reactor_type == "Laminar Flame Reactor":
                results = comparison.compare_mechanisms(condition, reactor_type="premixed_flame")
            else:  # Constant Pressure Reactor
                results = comparison.compare_mechanisms(condition, reactor_type="constant_pressure")
            
            if results is None:
                raise ValueError("Simulation failed for one or both mechanisms.")
            
            # Display results
            output = "=== Flame Simulation Results ===\n\n"
            output += f"Reactor Type: {reactor_type}\n"
            output += f"Burning Velocity (Reduced): {results['reduced_burning_velocity']:.4f} m/s\n"
            output += f"Burning Velocity (GRI-Mech): {results['gri_burning_velocity']:.4f} m/s\n"
            output += f"Max Temperature (Reduced): {results['reduced_max_temp']:.2f} K\n"
            output += f"Max Temperature (GRI-Mech): {results['gri_max_temp']:.2f} K\n"
            output += "\nPlots saved as:\n"
            output += "- temperature_profile.png\n"
            output += "- mole_fraction_profiles.png\n"
            output += "- heat_release_rate.png\n"
            
            self.flame_output_text.insert(tk.END, output)
            self.flame_output_text.insert(tk.END, "\nFlame simulation completed.\n")
        except Exception as e:
            self.flame_output_text.insert(tk.END, f"Error running flame simulation: {str(e)}\n")
            messagebox.showerror("Flame Simulation Error", f"An error occurred: {str(e)}")
    
    def run_const_pressure_simulation(self):
        self.const_pressure_output_text.delete(1.0, tk.END)
        self.const_pressure_output_text.insert(tk.END, "Running constant pressure simulation...\n")
        self.root.update()
        
        try:
            # Validate inputs
            temperature = float(self.const_pressure_vars["temperature"].get())
            pressure = float(self.const_pressure_vars["pressure"].get())
            equivalence_ratios = json.loads(self.const_pressure_vars["equivalence_ratios"].get())
            fuel = self.const_pressure_vars["fuel"].get()
            oxidizer_str = self.const_pressure_vars["oxidizer"].get()
            end_time = float(self.const_pressure_vars["end_time"].get())
            key_species = json.loads(self.const_pressure_vars["key_species"].get())
            reduced_mech_path = self.const_pressure_vars["reduced_mech_path"].get()
            full_mech_name = self.const_pressure_vars["full_mech_name"].get()
            output_dir = self.const_pressure_vars["output_dir"].get()
            xlim = float(self.const_pressure_vars["xlim"].get())
            
            if not reduced_mech_path or not os.path.exists(reduced_mech_path):
                raise ValueError("Invalid or missing reduced mechanism file")
            if not full_mech_name:
                raise ValueError("Invalid full mechanism name")
            try:
                ct.Solution(full_mech_name)
            except Exception as e:
                raise ValueError(f"Invalid full mechanism name: {str(e)}")
            if not output_dir:
                raise ValueError("Output directory must be specified")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if temperature <= 0:
                raise ValueError("Temperature must be positive")
            if pressure <= 0:
                raise ValueError("Pressure must be positive")
            if not equivalence_ratios:
                raise ValueError("Equivalence ratios list cannot be empty")
            if any(phi <= 0 for phi in equivalence_ratios):
                raise ValueError("Equivalence ratios must be positive")
            if end_time <= 0:
                raise ValueError("End time must be positive")
            if not key_species:
                raise ValueError("Key species list cannot be empty")
            if xlim <= 0:
                raise ValueError("X-axis limit must be positive")
            
            # Parse oxidizer
            oxidizer_dict = {}
            for pair in oxidizer_str.split(","):
                species, ratio = pair.strip().split(":")
                oxidizer_dict[species] = float(ratio)
            
            # Create conditions dictionary
            conditions = {
                "temperature": temperature,
                "pressure": pressure,
                "equivalence_ratios": equivalence_ratios,
                "fuel": fuel,
                "oxidizer": oxidizer_dict,
                "end_time": end_time,
                "key_species": key_species
            }
            
            # Initialize and run comparison
            comparison = MechanismComparison(output_dir, reduced_mech_path, full_mech_name, xlim)
            results = comparison.compare_mechanisms(conditions)
            
            if results is None:
                raise ValueError("Simulation failed for one or more equivalence ratios.")
            
            # Display results
            output = "=== Constant Pressure Simulation Results ===\n\n"
            for phi, idts in results["idts"].items():
                output += f"Equivalence Ratio (phi): {phi}\n"
                output += f"IDT (Reduced): {idts['reduced']:.6f} ms\n"
                output += f"IDT (Full): {idts['full']:.6f} ms\n"
                output += f"Max Temperature (Reduced): {results['max_temps'][phi]['reduced']:.2f} K\n"
                output += f"Max Temperature (Full): {results['max_temps'][phi]['full']:.2f} K\n\n"
            
            output += "Plots saved in the output directory:\n"
            output += "- IDT.png\n"
            output += "- temperature.png\n"
            for species in key_species:
                output += f"- mole_fraction_{species}.png\n"
            
            self.const_pressure_output_text.insert(tk.END, output)
            self.const_pressure_output_text.insert(tk.END, "\nConstant pressure simulation completed.\n")
        except Exception as e:
            self.const_pressure_output_text.insert(tk.END, f"Error running constant pressure simulation: {str(e)}\n")
            messagebox.showerror("Constant Pressure Simulation Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()