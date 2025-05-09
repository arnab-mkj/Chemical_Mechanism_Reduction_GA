import os
import json
import numpy as np


def save_mole_fractions_to_json(results, species_names, generation, filename="outputs/mole_fractions.json"):
    """
    Save mole fractions to a JSON file.

    Parameters:
        results (dict): Simulation results containing mole fractions.
        species_names (list): List of species names in the mechanism.
        generation (int): Current generation number.
        filename (str): Name of the JSON file.
    """
    try:
        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Extract mole fractions
        mole_fractions = results.get("mole_fractions", {})
        if not isinstance(mole_fractions, (dict, np.ndarray)):
            raise TypeError("Mole fractions must be a dictionary or numpy.ndarray")

        # Convert numpy types to Python-native types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            else:
                return obj

        # Prepare data for saving
        data = {
            "generation": generation,
            "species": species_names,
            "mole_fraction": convert_numpy_types(mole_fractions),
            "temperature": results.get("temperature", None),
            "pressure": results.get("pressure", None)
        }

        # Append to existing data or create a new file
        if os.path.exists(filename):
            try:
                with open(filename, "r") as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Warning: {filename} is corrupted. Overwriting the file.")
                existing_data = []
            existing_data.append(data)
        else:
            existing_data = [data]

        with open(filename, "w") as file:
            json.dump(existing_data, file, indent=4)

        print(f"Mole fractions saved to: {filename}")

    except (IOError, ValueError, TypeError) as e:
        print(f"Error while saving mole fractions to JSON: {e}")
        raise e
    
    
def save_species_concentrations(results, species_names, generation, filename="species_concentrations.json"):
    """
    Save species concentrations for selected species over generations.

    Parameters:
        results (dict): Simulation results containing mole fractions.
        species_names (list): List of species names in the mechanism.
        generation (int): Current generation number.
        filename (str): Name of the JSON file to save species concentrations.
    """
    
    try:
        # Ensure the directory for the file exists
        print(f"Saving species concentrations to: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f"Directory created or already exists: {os.path.dirname(filename)}")
        except Exception as e:
            print(f"Error creating directory: {e}")

        # Selected species to track
        selected_species = ["CO2", "H2O", "CH4"]  # Add or modify species as needed

        # Extract mole fractions for selected species
        mole_fractions = results.get("mole_fractions", [])
        species_data = {
            "generation": generation,
            "species_concentrations": {
                species: mole_fractions[species_names.index(species)] if species in species_names else 0.0
                for species in selected_species
            },
            "temperature": results.get("temperature", None),
            "pressure": results.get("pressure", None)
        }

        # Save to JSON file
        if os.path.exists(filename):
            with open(filename, "r") as file:
                existing_data = json.load(file)
            existing_data.append(species_data)
        else:
            existing_data = [species_data]

        with open(filename, "w") as file:
            json.dump(existing_data, file, indent=4)

        print(f"Species concentrations saved to: {filename}")

    except Exception as e:
        print(f"Error while saving species concentrations: {e}")
        raise e