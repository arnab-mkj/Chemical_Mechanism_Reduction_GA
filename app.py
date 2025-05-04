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

class GeneticAlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm for Chemical Mechanism Reduction")
        self.root.geometry("1000x800")
        
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.main_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_frame, text="Configuration")
        
        self.tests_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tests_frame, text="Tests")
        
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
        
        self.test_vars = {
            # Common parameters
            "full_mech": tk.StringVar(value="gri30.yaml"),
            "reduced_mech": tk.StringVar(value="reduced_random_gri30.yaml"),
            "reduction_fraction": tk.StringVar(value="0.3"),
            # Mutation parameters
            "mutation_rate": tk.StringVar(value="0.01"),
            # Fitness parameters
            "temperature": tk.StringVar(value="1800"),
            "pressure": tk.StringVar(value="1e5"),
            "equivalence_ratio": tk.StringVar(value="1.0"),
            "fuel": tk.StringVar(value="CH4"),
            "oxidizer": tk.StringVar(value="O2:0.21, N2:0.79"),
            # Initialization parameters
            "diversity_prob": tk.StringVar(value="0.05"),
            "remove_reactions": tk.StringVar(value='["H + O2 <=> O + OH", "HO2 + OH <=> H2O + O2"]')
        }
        
        self.create_input_fields()
        self.create_test_interface()
        
        self.output_text = tk.Text(self.main_frame, height=10, width=80)
        self.output_text.grid(row=len(self.config_vars) + 6, column=0, columnspan=3, pady=10)
        
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
        
        for key, var in self.config_vars.items():
            if key in ["species_weights_input", "weights_method", "elitism_enabled", "deactivation_chance", "init_with_reduced_mech", "elite_size", "reduction_threshold"]:
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
        
        # Placeholder frame for test-specific fields
        self.test_fields_frame = ttk.Frame(self.tests_frame)
        self.test_fields_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        self.output_text = tk.Text(self.tests_frame, height=10, width=80)
        self.output_text.grid(row=row, column=0, columnspan=3, pady=10)
        
        # Initialize fields for the default test (mutation)
        self.update_test_fields("mutation")
    
    def update_test_fields(self, test_type):
        # Clear previous fields
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
    
    def browse_file(self, key):
        path = filedialog.askdirectory(title=f"Select {key.replace('_', ' ').title()}")
        if path:
            self.config_vars[key].set(path)
    
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
                # Prepare parameters for params_test.json
                full_mech = self.test_vars["full_mech"].get()
                reduced_mech = self.test_vars["reduced_mech"].get()
                reduction_fraction = float(self.test_vars["reduction_fraction"].get())
                temperature = float(self.test_vars["temperature"].get())
                pressure = float(self.test_vars["pressure"].get())
                equivalence_ratio = float(self.test_vars["equivalence_ratio"].get())
                fuel_str = self.test_vars["fuel"].get()
                oxidizer_str = self.test_vars["oxidizer"].get()
                
                # Parse oxidizer composition
                oxidizer_dict = {}
                for pair in oxidizer_str.split(","):
                    species, ratio = pair.strip().split(":")
                    oxidizer_dict[species] = float(ratio)
                
                # Get key_species from main config
                key_species = json.loads(self.config_vars["key_species"].get())
                
                # Prepare params_test.json
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
            
            else:
                # For crossover and selection, which don't have specific parameters yet
                result = module.run_test()
            
            self.output_text.insert(tk.END, f"Test {self.test_var.get()} completed.\nResult: {result}\n")
        except Exception as e:
            self.output_text.insert(tk.END, f"Error running test {self.test_var.get()}: {str(e)}\n")
            messagebox.showerror("Test Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmGUI(root)
    root.mainloop()