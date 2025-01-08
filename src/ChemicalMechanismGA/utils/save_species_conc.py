import os
import json
import numpy as np


def save_mole_fractions_to_json(results, species_names, generation, filename="mole_fractions.json"):
    """
    Save mole fractions to a CSV file.
    
    Parameters:
        mole_fractions (dict): Mole fractions of species.
        generation (int): Current generation number.
        filename (str): Name of the CSV file.
    """
    try:
        # Ensure the directory for the file exists
        filename = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/output/mole_fractions.json"
        print(f"Saving mole fractions to: {filename}")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f"Directory created or already exists: {os.path.dirname(filename)}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            
        mole_fractions = results.get("mole_fractions", None) # extract the mole fractions
        if mole_fractions is None:
            raise ValueError("Mole fractions are missing in the results")
        
        # check if mole fractions are a numpy array
        #if not isinstance(mole_fractions, np.ndarray):
            #raise TypeError("Mole fractiosn must be a numpy.ndarray")
            
        data ={
            "generation": generation,
            "species": species_names,
            "mole_fraction": mole_fractions.tolist() if isinstance(mole_fractions, np.ndarray)
            else mole_fractions,
            "temperature": results.get("temperature", None),
            "pressure": results.get("pressure", None)
        }
            
        # save mole fractions to json
        if os.path.exists(filename):
            with open(filename, "r") as file:
                existing_data = json.load(file)
            existing_data.append(data)
        else:
            existing_data = [data]
        
        with open(filename, "w") as file:
            json.dump(existing_data, file, indent=4)
        
    except Exception as e:
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
        filename = "E:/PPP_WS2024-25/ChemicalMechanismReduction/data/output/mole_fractions.json"
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