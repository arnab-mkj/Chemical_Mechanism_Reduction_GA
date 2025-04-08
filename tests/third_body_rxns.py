import cantera as ct
import yaml

def get_third_body_reactions(mechanism_file):
    # Load the YAML file directly
    with open(mechanism_file, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Load the mechanism in Cantera for reference
    gas = ct.Solution('gri30.yaml')

    # Dictionary to store third-body reactions
    third_body_rxns = {}

    # Check if the YAML file has a 'reactions' section
    if 'reactions' in yaml_data:
        for i, reaction in enumerate(yaml_data['reactions']):
            # Check if the reaction is a three-body reaction by its type
            if 'type' in reaction and reaction['type'].lower() in ['three-body', 'threebody']:
                # Get the reaction equation from Cantera for better readability
                cantera_reaction = gas.reaction(i)

                # Get third-body efficiencies if available
                efficiencies = reaction.get('efficiencies', {})

                # Store the reaction index, equation, and third-body species with their efficiencies
                third_body_rxns[i] = {
                    'equation': cantera_reaction.equation,
                    'efficiencies': efficiencies
                }

    return third_body_rxns

# Example usage
mechanism_file = "data\gri30.yaml"  # Replace with your mechanism file
third_body_reactions = get_third_body_reactions(mechanism_file)

# Print the results
print("Third-Body Reactions and Associated Species with Efficiencies:")
for reaction_index, data in third_body_reactions.items():
    print(f"Reaction Index: {reaction_index}")
    print(f"Equation: {data['equation']}")
    print(f"Third-Body Species and Efficiencies: {data['efficiencies']}")
    print("-" * 50)