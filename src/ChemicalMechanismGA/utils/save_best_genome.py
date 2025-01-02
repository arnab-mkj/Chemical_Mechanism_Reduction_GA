import cantera as ct
import yaml
from yaml import YAMLObject

class ReactionYAMLObject(YAMLObject):
    """
    Custom YAMLObject for serializing Cantera Reaction objects.
    """
    yaml_tag = u'!Reaction'
    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper

    def __init__(self, reaction):
        self.reaction = reaction

    @classmethod
    def from_yaml(cls, loader, node):
        # Not needed for this use case, but can be implemented if deserialization is required
        raise NotImplementedError("Deserialization from YAML is not implemented.")

    @classmethod
    def to_yaml(cls, dumper, data):
        # Serialize the reaction to a YAML string
        return dumper.represent_scalar(cls.yaml_tag, data.reaction.to_yaml_string())

def save_genome_as_yaml(genome, original_mechanism_path, output_path):
    """
    Save the best genome as a reduced mechanism in YAML format.

    Args:
        genome (list): The binary genome indicating active (1) or inactive (0) reactions.
        original_mechanism_path (str): Path to the original GRI-Mech 3.0 YAML file.
        output_path (str): Path to save the reduced mechanism YAML file.
    """
    # Load the original mechanism
    gas = ct.Solution(original_mechanism_path)
    reactions = gas.reactions()

    # Filter reactions based on the genome
    reduced_reactions = [reaction for i, reaction in enumerate(reactions) if genome[i] == 1]

    # Create a new Cantera Solution object with the reduced mechanism
    reduced_gas = ct.Solution(
        thermo="IdealGas",
        kinetics="GasKinetics",
        species=gas.species(),
        reactions=reduced_reactions
    )

    # Convert the reduced mechanism to YAML format
    reduced_mechanism = {
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": reduced_gas.element_names,
                "species": [species.name for species in reduced_gas.species()],
                "reactions": [reaction.to_cti() for reaction in reduced_reactions],  # Use to_cti() as a fallback
            }
        ]
    }

    # Save the reduced mechanism to a YAML file
    with open(output_path, "w") as yaml_file:
        yaml.dump(reduced_mechanism, yaml_file, default_flow_style=False)

    print(f"Reduced mechanism saved to {output_path}")