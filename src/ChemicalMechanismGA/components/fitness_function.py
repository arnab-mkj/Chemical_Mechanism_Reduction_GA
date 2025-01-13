import cantera as ct
import csv
from .simulation_runner import SimulationRunner
import os
import numpy as np
import json
from ..utils.save_species_conc import save_mole_fractions_to_json, save_species_concentrations
#from ..utils.hyperparameter_tuning import HyperparameterTuner


def create_reduced_mechanism(genome, original_mechanism_path="gri30.yaml"):
    """
    Create a reduced mechanism based on the genome.
    
    Parameters:
        genome (list): Binary genome representing active reactions.
        original_mechanism_path (str): Path to the original mechanism file.
    
    Returns:
        ct.Solution: Reduced mechanism as a Cantera Solution object.
    """
    gas = ct.Solution(original_mechanism_path)
    reactions = gas.reactions()
    reduced_reactions = [
        reaction for i, reaction in enumerate(reactions) if genome[i] == 1
    ]
    
    if len(reduced_reactions) < 50:  # Arbitrary threshold
        raise ValueError("Reduced mechanism has too few reactions")
    
    reduced_mechanism = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=gas.species(),
        reactions=reduced_reactions,
    )
    print(f"Reduced mechanism created with {len(reduced_reactions)} reactions")
    return reduced_mechanism


def run_simulation_with_reduced_mechanism(reduced_mechanism, reactor_type="constant_pressure", generation=None):
    """
    Run a simulation using the reduced mechanism and save mole fractions.
    
    Parameters:
        reduced_mechanism (ct.Solution): Reduced mechanism as a Cantera Solution object.
        reactor_type (str): Type of reactor to use (e.g., "batch", "const_pressure").
        generation (int): Current generation number (used for saving results).
    
    Returns:
        dict: Simulation results (e.g., temperature, species mole fractions).
    """
    # Step 1: Initialize the SimulationRunner
    try:
        runner = SimulationRunner(mechanism_path=None, reactor_type=reactor_type)
        runner.gas = reduced_mechanism  # Use the reduced mechanism directly

        # Step 2: Set initial conditions
        initial_temperature = 1000.0  # Initial temperature in Kelvin
        initial_pressure = ct.one_atm  # Initial pressure in Pascals
        initial_species = {"CH4": 1.0, "O2": 2.0, "N2": 7.52}  # Stoichiometric mixture
        runner.set_initial_conditions(initial_temperature, initial_pressure, initial_species)
        print(f"Initial conditions set: T={initial_temperature}, P={initial_pressure}, X={initial_species}")
        
        # Step 3: Run the simulation
        runner.run_simulation(end_time=1.0, time_step=1e-5)

        # Step 4: Extract results
        results = runner.get_results()
        results["species_names"] = reduced_mechanism.species_names
        
        print(f"Simulation results: {results}")
        # Step nope: Save mole fractions to a CSV file
        #save_mole_fractions_to_json(results, reduced_mechanism.species_names, generation)

        return results
    except Exception as e:
        print(f"Error during simulation with reduced mechanism: {e}")
        raise


def evaluate_fitness(genome, original_mechanism_path="gri30.yaml", 
                     reactor_type="constant_pressure", 
                     generation=None,
                     filename=None):
    """Evaluate the fitness of a genome by running a simulation with the reduced mechanism.
    
    Parameters:
        genome (list): Binary genome representing active reactions.
        original_mechanism_path (str): Path to the original mechanism file.
        reactor_type (str): Type of reactor to use (e.g., "batch", "const_pressure").
        generation (int): Current generation number (used for saving results).
    
    Returns:
        float: Fitness score (lower is better).
    """
    try:
        # Step 1: Create the reduced mechanism
        reduced_mechanism = create_reduced_mechanism(genome, original_mechanism_path)

        # Step 2: Run the simulation
        results = run_simulation_with_reduced_mechanism(reduced_mechanism, reactor_type, generation)

        print(f"Simulaiton results for Generation {generation}: {results}")
        
        # step 3: check for mole fractions
        if "mole_fractions" not in results:
            raise KeyError("Mole fractions are missing from the simulation results")
        
        # step 4: Initialize the fitness evaluator
        fitness_evaluator = FitnessEvaluator(
           target_temperature=2000.0,
            target_species={"CO2": 0.1, "H2O": 0.2}, #example species, change as necessary
            target_delay=1.0,
            weight_temperature=1.0,
            weight_species=1.5,
            weight_ignition_delay=0.5,
            difference_function="squared" 
        )
        fitness = fitness_evaluator.combined_fitness(results)
        print(f"Fitness Score for Generation {generation}: {fitness}")
        return fitness, results

    except Exception as e:
        print(f"Error during fitness evaluation for genome {genome}: {e}")
        return 1e6, None  # Penalize invalid solutions
    
def run_generation(population, original_mechanism_path, 
                   reactor_type, 
                   generation,
                   filename="mole_fractions.json", 
                   species_filename="species_concentrations.json"):
    """
    Run a single generation of the genetic algorithm

    Args:
        population (list): List of genomes in the current generation
        original_mechanism_path (str): Path to the original mechanism path
        reactor_type (str): Type of reactor to be used
        generation (int): current generation number
        filename (str, optional):  Defaults to "mole_fractions.json".

    Returns:
        _type_: list: Fitness scores of the current generation
    """

    fitness_scores = []
    best_fitness = float("inf")
    best_results = None
    best_species_names = None
    
    for genome in population:
        fitness, results = evaluate_fitness(
                            genome,
                            original_mechanism_path=original_mechanism_path,
                            reactor_type=reactor_type,
                            generation=generation,
                            filename=None)
        
        fitness_scores.append(fitness)
        
        # Update best results if this genome is better
        if fitness < best_fitness:
            best_fitness = fitness

            best_results = results # results from evaluate fitness
            
            best_species_names = results["species_names"]
            
     # save the mole fractions for the best genome of the generation and validation
    if best_results is not None and "mole_fractions" in best_results:
        save_mole_fractions_to_json(best_results, best_species_names, generation, filename) 
        # Save species concentrations for selected species
        save_species_concentrations(best_results, best_species_names, generation, species_filename)
    
    return fitness_scores
        
def validate_reduced_mechanism(reduced_mechanism):
    if len(reduced_mechanism.reactions()) < 50:  # Arbitrary threshold
        raise ValueError("Reduced mechanism has too few reactions")   
    
class FitnessEvaluator:
    def __init__(self, target_temperature=2000.0, target_species=None, target_delay=1.0,
                 weight_temperature=1.0, weight_species=1.0, weight_ignition_delay=1.0,
                 difference_function="absolute"):
        """
        Initialize the FitnessEvaluator with target values.

        Parameters:
            target_temperature (float): Target temperature for fitness evaluation.
            target_species (dict): Target mole fractions for key species (e.g., {"CO2": 0.1, "H2O": 0.2}).
        """
        self.target_temperature = target_temperature
        self.target_species = target_species if target_species else {}
        self.target_delay= target_delay
        self.weight_temperature = weight_temperature
        self.weight_species = weight_species
        self.weight_ignition_delay = weight_ignition_delay
        self.difference_function = difference_function
        
    def calculate_difference(self, actual, target):
        if self.difference_function == "absolute":
            return abs(actual - target)
        elif self.difference_function == "squared":
            return (actual - target) ** 2
        elif self.difference_function == "relative":
            return abs((actual - target) / target) if target != 0 else float("inf")
        else:
            raise ValueError(f"Unsupported difference function: {self.difference_function}")

    def temperature_fitness(self, results):
        """
        Calculate fitness based on the difference between the actual and target temperature.

        Parameters:
            results (dict): Simulation results containing temperature.

        Returns:
            float: Fitness score (lower is better).
        """
        actual_temperature = results.get("temperature", 0.0)
        fitness = abs(actual_temperature - self.target_temperature)
        print(f"Temperature Fitness: {fitness} (Actual: {actual_temperature}, Target: {self.target_temperature})")
        return fitness

    def species_fitness(self, results):
        """
        Calculate fitness based on the difference between actual and target species mole fractions.

        Parameters:
            results (dict): Simulation results containing mole fractions.

        Returns:
            float: Fitness score (lower is better).
        """
        mole_fractions = results.get("mole_fractions", None)
        if mole_fractions is None:
            raise ValueError("Mole fractions missing in the results")
        
        species_name_to_index = {name: i for i, name in enumerate(results["species_names"])}
        
        fitness = 0.0
        for species, target_fraction in self.target_species.items():
            if species not in species_name_to_index:
                print(f"Warning! species {species} not found in the mechanism.")
                actual_fraction = 0.0
            else:
                actual_fraction = mole_fractions[species_name_to_index[species]]
                
            fitness += self.calculate_difference(actual_fraction, target_fraction)
            print(f"Species Fitness for {species}: {self.calculate_difference(
                actual_fraction, target_fraction)} (Actual: {actual_fraction}, Target: {target_fraction})")
        fitness /= len(self.target_species)
        return fitness

    
    def ignition_delay_fitness(self, results):
        """
    Calculate fitness based on the difference between actual and target ignition delay time.

    Parameters:
        results (dict): Simulation results containing ignition delay time.
        target_delay (float): Target ignition delay time (in seconds).

    Returns:
        float: Fitness score (lower is better).
    """
        actual_delay = results.get("ignition_delay", 0.0)
        fitness = self.calculate_difference(actual_delay, self.target_delay)
        print(f"Ignition Delay Fitness: {fitness} (Actual: {actual_delay}, Target: {self.target_delay})")
        return fitness
    
    def combined_fitness(self, results):
        """
        Combine temperature and species fitness into a single score.

        Parameters:
            results (dict): Simulation results containing temperature and mole fractions.

        Returns:
            float: Combined fitness score (lower is better).
        """
        temp_fitness = self.temperature_fitness(results) * self.weight_temperature
        species_fitness = self.species_fitness(results) * self.weight_species
        ignition_fitness = self.ignition_delay_fitness(results) * self.weight_ignition_delay
    
        total_fitness = temp_fitness + species_fitness + ignition_fitness
        print(f"Combined Fitness: {total_fitness}")
        return total_fitness