import cantera as ct
import yaml
import os

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

    # Manually serialize reactions to YAML format
    serialized_reactions = []
    for reaction in reduced_reactions:
        # Serialize the reaction manually
        reaction_data = {
            "equation": reaction.equation,
            "type": reaction.reaction_type,
        }

        # Handle different rate types
        if isinstance(reaction.rate, ct.ArrheniusRate):
            reaction_data["rate-constant"] = {
                "A": reaction.rate.pre_exponential_factor,
                "b": reaction.rate.temperature_exponent,
                "Ea": reaction.rate.activation_energy / 4184.0,  # Convert J/mol to kcal/mol
            }
        elif isinstance(reaction.rate, ct.LindemannRate):
            reaction_data["low-P-rate-constant"] = {
                "A": reaction.rate.low_rate.pre_exponential_factor,
                "b": reaction.rate.low_rate.temperature_exponent,
                "Ea": reaction.rate.low_rate.activation_energy / 4184.0,
            }
            if reaction.rate.high_rate:
                reaction_data["high-P-rate-constant"] = {
                    "A": reaction.rate.high_rate.pre_exponential_factor,
                    "b": reaction.rate.high_rate.temperature_exponent,
                    "Ea": reaction.rate.high_rate.activation_energy / 4184.0,
                }
        elif isinstance(reaction.rate, ct.TroeRate):
            reaction_data["low-P-rate-constant"] = {
                "A": reaction.rate.low_rate.pre_exponential_factor,
                "b": reaction.rate.low_rate.temperature_exponent,
                "Ea": reaction.rate.low_rate.activation_energy / 4184.0,
            }
            reaction_data["high-P-rate-constant"] = {
                "A": reaction.rate.high_rate.pre_exponential_factor,
                "b": reaction.rate.high_rate.temperature_exponent,
                "Ea": reaction.rate.high_rate.activation_energy / 4184.0,
            }
            reaction_data["Troe"] = {
                "A": reaction.rate.falloff_coeffs[0],
                "T3": reaction.rate.falloff_coeffs[1],
                "T1": reaction.rate.falloff_coeffs[2],
                "T2": reaction.rate.falloff_coeffs[3] if len(reaction.rate.falloff_coeffs) > 3 else None,
            }
        else:
            reaction_data["rate-constant"] = "Unsupported rate type"

        serialized_reactions.append(reaction_data)

    # Convert the reduced mechanism to YAML format
    reduced_mechanism = {
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": reduced_gas.element_names,
                "species": [species.name for species in reduced_gas.species()],
                "reactions": serialized_reactions,
            }
        ]
    }

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the reduced mechanism to a YAML file
    with open(output_path, "w") as yaml_file:
        yaml.dump(reduced_mechanism, yaml_file, default_flow_style=False)

