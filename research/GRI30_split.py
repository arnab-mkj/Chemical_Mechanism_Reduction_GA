import cantera as ct
import yaml
from collections import defaultdict
from pathlib import Path
import re

class SpeciesManager:
    def __init__(self, mechanism_data=None):
        """
        Initialize the SpeciesManager with mechanism data.

        Args:
            mechanism_data (dict): Dictionary containing species and reactions data
        """
        self.species_list = mechanism_data['species'] if mechanism_data else []
        self.reactions_list = mechanism_data['reactions'] if mechanism_data else []
        self.species_data = {}

    def save_species_data(self, output_dir='./data'):
        """
        Save the species list to a YAML file.

        Args:
            output_dir (str): Directory to save the output file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        species_file = {
            'species_count': len(self.species_list),
            'species': self.species_list
        }

        output_file = f'{output_dir}/species_data.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(species_file, f, default_flow_style=False)
        print(f"Species data saved to {output_file}")
        return species_file



class MechanismAnalyzer:
    def __init__(self, mechanism_name='gri30.yaml'):
        """
        Initialize the MechanismAnalyzer with a mechanism file.

        Args:
            mechanism_name (str): Name of the mechanism file (default: 'gri30.yaml')
        """
        self.mechanism_name = mechanism_name
        self.gas = ct.Solution(mechanism_name)
        self.species_manager = None
        self.mechanism_data = None



    def extract_mechanism_data(self):
        """Extract species and reactions from the mechanism."""
        # Extract species and reactions
        species_list = self.gas.species_names
        reactions_list = [rxn.equation for rxn in self.gas.reactions()]

        # Create dictionary for YAML
        self.mechanism_data = {
            'species': species_list,
            'reactions': reactions_list
        }
        
        self.species_manager = SpeciesManager(self.mechanism_data)
        # Save to YAML file
        output_dir = './data'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = f'{output_dir}/mechanism_data.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(self.mechanism_data, f, default_flow_style=False)

       
        self.species_manager.save_species_data()
        print(f"Mechanism data saved to {output_file}")
        return self.mechanism_data

    def split_reaction_equation(self, equation):
    
        if '<=>' in equation:
            return equation.split('<=>')[0].strip()
        elif '=>' in equation:
            return equation.split('=>')[0].strip()
        elif '<=' in equation:
            return equation.split('<=')[0].strip()
        return equation.strip()

    def analyze_all_species(self):
        """
        Analyze reactions for all species and save in a single file.
        """
        if not self.species_manager:
            raise ValueError("Please run extract_mechanism_data() first")

        all_species_analysis = {}
        # Analyze each species
        for species in self.species_manager.species_list:
            occurrence_count = 0
            reaction_indices = []
            pattern = r'\b' + re.escape(species) + r'\b'
            # Check each reaction for the species
            for idx, reaction in enumerate(self.gas.reactions()):
                reactants_part = self.split_reaction_equation(reaction.equation)
                if re.search(pattern, reactants_part):
                    occurrence_count += 1
                    reaction_indices.append(idx)
                    
            # Store analysis for this species
            all_species_analysis[species] = {
                'occurrence_count': occurrence_count,
                'reactions': [self.mechanism_data['reactions'][i] for i in reaction_indices]
            }

        # Save complete analysis to YAML file
        output_dir = './data'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = f'{output_dir}/complete_species_analysis.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(all_species_analysis, f, default_flow_style=False)
        print(f"Complete species analysis saved to {output_file}")
        return all_species_analysis



if __name__ == "__main__":
    # Create analyzer instance
    analyzer = MechanismAnalyzer()

    mechanism_data = analyzer.extract_mechanism_data()
    print(f"Total species: {len(mechanism_data['species'])}")
    print(f"Total reactions: {len(mechanism_data['reactions'])}")
   
    complete_analysis = analyzer.analyze_all_species()

    print("\nFiles created during execution:")
    print("- data/mechanism_data.yaml")
    print("- data/species_data.yaml")
    print("- data/complete_species_analysis.yaml")