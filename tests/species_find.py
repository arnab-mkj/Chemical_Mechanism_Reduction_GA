import cantera as ct
import yaml
import json

def find_species_in_reduced_genome(original_mechanism_file, genome_file):
    """
    Find the number of species involved in the reactions corresponding to the reduced genome.

    Parameters:
    -----------
    original_mechanism_file : str
        Path to the original mechanism file (YAML format).
    genome_file : str
        Path to the JSON file containing the reduced genome.

    Returns:
    --------
    list
        A list of unique species involved in the active reactions.
    """
    # Load the reduced genome
    with open(genome_file, 'r') as f:
        genome_data = json.load(f)

    genome = genome_data['genome']
    active_reactions = sum(genome)

    print(f"Loaded genome with {active_reactions} active reactions out of {len(genome)} total reactions.")

    # Load the original mechanism
    with open(original_mechanism_file, 'r') as f:
        mechanism_data = yaml.safe_load(f)

    # Load the mechanism in Cantera for reference
    gas = ct.Solution(original_mechanism_file)

    # Get the original reactions
    original_reactions = mechanism_data['reactions']

    # Filter active reactions based on the genome
    active_reaction_indices = [i for i, active in enumerate(genome) if active == 1]
    active_reactions = [original_reactions[i] for i in active_reaction_indices]

    # Find all species involved in the active reactions
    involved_species = set()

    for i in active_reaction_indices:
        reaction = gas.reaction(i)

        # Add reactants and products
        involved_species.update(reaction.reactants.keys())
        involved_species.update(reaction.products.keys())

        # Add third-body species (if applicable)
        if hasattr(reaction, 'efficiencies'):
            involved_species.update(reaction.efficiencies.keys())

    # Convert to a sorted list for readability
    involved_species = sorted(involved_species)

    print(f"Number of unique species involved: {len(involved_species)}")
    print("List of species involved:")
    for species in involved_species:
        print(f"- {species}")

    return involved_species

# Example usage
original_mechanism = "data\gri30.yaml"  # Replace with your original mechanism file
genome_file = "best_genome_1.json"  # The file saved from the previous analysis

involved_species = find_species_in_reduced_genome(original_mechanism, genome_file)