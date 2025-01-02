import cantera as ct
import csv
from .simulation_runner import SimulationRunner
import os
import numpy as np


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
    reduced_mechanism = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=gas.species(),
        reactions=reduced_reactions,
    )
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
    runner = SimulationRunner(mechanism_path=None, reactor_type=reactor_type)
    runner.gas = reduced_mechanism  # Use the reduced mechanism directly

    # Step 2: Set initial conditions
    initial_temperature = 1000.0  # Initial temperature in Kelvin
    initial_pressure = ct.one_atm  # Initial pressure in Pascals
    initial_species = {"CH4": 1.0, "O2": 3.0, "N2": 3.76}  # Stoichiometric mixture
    runner.set_initial_conditions(initial_temperature, initial_pressure, initial_species)

    # Step 3: Run the simulation
    runner.run_simulation(end_time=10.0, time_step=0.5)

    # Step 4: Extract results
    results = runner.get_results()

    # Step 5: Save mole fractions to a CSV file
    save_mole_fractions_to_csv(results, reduced_mechanism.species_names, generation)

    return results


def save_mole_fractions_to_csv(results, species_names, generation, filename="mole_fractions.csv"):
    """
    Save mole fractions to a CSV file.
    
    Parameters:
        mole_fractions (dict): Mole fractions of species.
        generation (int): Current generation number.
        filename (str): Name of the CSV file.
    """
    try:
        # Ensure the directory for the file exists
        filename = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/output/mole_fractions.csv"
        print(f"Saving mole fractions to: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f"Directory created or already exists: {os.path.dirname(filename)}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            
        mole_fractions = results.get("mole_fractions", None)
        if mole_fractions is None:
            raise ValueError("Mole fractions are missing in the results")
        
        # check if mole fractions are a numpy array
        if not isinstance(mole_fractions, np.ndarray):
            raise TypeError("Mole fractiosn must be a numpy.ndarray")
            
        # save mole fractions to csv
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if generation == 0:  # Write header only for the first generation
                writer.writerow(["Generation"] + species_names)
               
            # Write mole fractions for the current generation
            for mf in mole_fractions:
                writer.writerow([generation] + list(mf))
            print(f"Mole fractions saved successfully for generation {generation}.")
    except Exception as e:
        print(f"Error while saving mole fractions: {e}")
        raise


def calculate_fitness(results, target_temperature=1200.0):
    """
    Calculate the fitness score based on simulation results.
    
    Parameters:
        results (dict): Simulation results (e.g., temperature, species mole fractions).
        target_temperature (float): Target temperature for fitness evaluation.
    
    Returns:
        float: Fitness score (lower is better).
    """
    # Example fitness calculation: minimize the difference from the target temperature
    fitness = abs(results["temperature"] - target_temperature)
    return fitness


def evaluate_fitness(genome, original_mechanism_path="gri30.yaml", 
                     reactor_type="constant_pressure", 
                     generation=None,
                     filename="mole_fraction.csv"):
    """
    Evaluate the fitness of a genome by running a simulation with the reduced mechanism.
    
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
        
        #  Calculate the fitness score
        #fitness = calculate_fitness(results)

        # step 4: Initialize the fitness evaluator
        fitness_evaluator = FitnessEvaluator(
            target_temperature = 2000.0,
            target_species={"CO2": 0.1, "H2O":0.2} #example species, change as necessary
        )
        fitness = fitness_evaluator.combined_fitness(results)
        print(f"Fitness Score for Generation {generation}: {fitness}")
        return fitness

    except Exception as e:
        print(f"Error during fitness evaluation: {e}")
        return 1e6  # Penalize invalid solutions
    
    

class FitnessEvaluator:
    def __init__(self, target_temperature=2000.0, target_species=None):
        """
        Initialize the FitnessEvaluator with target values.

        Parameters:
            target_temperature (float): Target temperature for fitness evaluation.
            target_species (dict): Target mole fractions for key species (e.g., {"CO2": 0.1, "H2O": 0.2}).
        """
        self.target_temperature = target_temperature
        self.target_species = target_species if target_species else {}

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
        mole_fractions = results.get("mole_fractions", {})
        fitness = 0.0
        for species, target_fraction in self.target_species.items():
            actual_fraction = mole_fractions.get(species, 0.0)
            fitness += abs(actual_fraction - target_fraction)
            print(f"Species Fitness for {species}: {abs(actual_fraction - target_fraction)} (Actual: {actual_fraction}, Target: {target_fraction})")
        return fitness

    def combined_fitness(self, results):
        """
        Combine temperature and species fitness into a single score.

        Parameters:
            results (dict): Simulation results containing temperature and mole fractions.

        Returns:
            float: Combined fitness score (lower is better).
        """
        temp_fitness = self.temperature_fitness(results)
        species_fitness = self.species_fitness(results)
        total_fitness = temp_fitness + species_fitness
        print(f"Combined Fitness: {total_fitness}")
        return total_fitness
    
    def ignition_delay_fitness(self, results, target_delay=1.0):
        """
    Calculate fitness based on the difference between actual and target ignition delay time.

    Parameters:
        results (dict): Simulation results containing ignition delay time.
        target_delay (float): Target ignition delay time (in seconds).

    Returns:
        float: Fitness score (lower is better).
    """
        actual_delay = results.get("ignition_delay", 0.0)
        fitness = abs(actual_delay - target_delay)
        print(f"Ignition Delay Fitness: {fitness} (Actual: {actual_delay}, Target: {target_delay})")
        return fitness