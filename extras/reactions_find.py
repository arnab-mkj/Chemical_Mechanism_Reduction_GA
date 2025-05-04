import cantera as ct
import json
import csv
from ruamel.yaml import YAML

def load_reactions_from_gri30():
    gas = ct.Solution('gri30.yaml')
    reactions = [str(r) for r in gas.reactions()]
    return reactions



def load_generation_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_best_genome(best_genome_file):
    with open(best_genome_file, 'r') as f:
        best_data = json.load(f)
    return best_data["genome"]

def track_permanently_removed_reactions(data, reaction_list, best_genome):
    num_reactions = len(reaction_list)
    max_gen = max(entry['generation'] for entry in data)

    last_active_gen = [None] * num_reactions
    data_sorted = sorted(data, key=lambda x: x['generation'])

    for entry in data_sorted:
        gen = entry['generation']
        genome = entry['overall_best_genome']
        for i in range(num_reactions):
            if genome[i] == 1:
                last_active_gen[i] = gen

    removed_reactions = []
    for i, last_gen in enumerate(last_active_gen):
        if best_genome[i] == 1:
            continue
        if last_gen is not None and last_gen < max_gen:
            removed_reactions.append({
                'index': i,
                'reaction': reaction_list[i],
                'generation_removed': last_gen + 1
            })

    return removed_reactions

def save_removed_reactions_to_csv(removed_reactions, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'reaction', 'generation_removed'])
        writer.writeheader()
        for row in removed_reactions:
            writer.writerow(row)


def extract_yaml_kinetics(yaml_file):
    """
    Parses the YAML file and returns a mapping of all reaction equation strings
    to a list of Arrhenius parameter dictionaries.
    Handles falloff and three-body reactions properly.
    """
    yaml = YAML()
    with open(yaml_file) as f:
        data = yaml.load(f)

    reaction_kinetics = {}

    for reaction in data['reactions']:
        eqn = reaction.get('equation', '')
        reaction_type = reaction.get('type', '')

        if reaction_type == 'falloff':
            # Store both high-P and low-P rates
            high = reaction.get('high-P-rate-constant', {})
            low = reaction.get('low-P-rate-constant', {})

            reaction_kinetics[eqn + " (high-P)"] = {
                'A': high.get('A', None),
                'b': high.get('b', None),
                'Ea': high.get('Ea', None)
            }
            reaction_kinetics[eqn + " (low-P)"] = {
                'A': low.get('A', None),
                'b': low.get('b', None),
                'Ea': low.get('Ea', None)
            }

        elif reaction_type == 'three-body':
            rate = reaction.get('rate-constant', {})
            reaction_kinetics[eqn + " (three-body)"] = {
                'A': rate.get('A', None),
                'b': rate.get('b', None),
                'Ea': rate.get('Ea', None)
            }

        else:
            rate = reaction.get('rate-constant', {})
            reaction_kinetics[eqn] = {
                'A': rate.get('A', None),
                'b': rate.get('b', None),
                'Ea': rate.get('Ea', None)
            }

    return reaction_kinetics


def save_active_reactions(best_genome, reduced_mech_yaml, csv_file):
    """
    Saves active reactions with their Arrhenius parameters (from YAML) into a CSV.
    A-values are formatted in scientific notation.
    """

    # Load full reaction list from Cantera
    full_gas = ct.Solution('gri30.yaml')
    full_reactions = [r.equation for r in full_gas.reactions()]  # Important: use r.equation not str(r)

    # Load YAML kinetics map
    yaml_kinetics_map = extract_yaml_kinetics(reduced_mech_yaml)

    # Collect only active reactions
    active_reactions = []
    for i, bit in enumerate(best_genome):
        if bit == 1:
            reaction_eqn = full_reactions[i]

            # Find kinetics (falloff types may be missing unless formatted correctly)
            # Try basic match first
            kinetics = yaml_kinetics_map.get(reaction_eqn)

            # If not found, try "(high-P)" or "(three-body)" suffixes
            if kinetics is None:
                for suffix in [" (high-P)", " (low-P)", " (three-body)"]:
                    kinetics = yaml_kinetics_map.get(reaction_eqn + suffix)
                    if kinetics:
                        reaction_eqn += suffix  # update label to match
                        break

            if kinetics is None:
                kinetics = {'A': None, 'b': None, 'Ea': None}

            # Format A in scientific notation
            A_val = kinetics['A']
            Ea_val = kinetics['Ea']
            if A_val is not None:
                A_val = f"{A_val:.3e}"
            if Ea_val is not None:
                Ea_val = f"{Ea_val/1000:.5e}"

            active_reactions.append({
                'index': i,
                'reaction': reaction_eqn,
                'A': A_val,
                'b': kinetics['b'],
                'Ea': Ea_val # in kcal/mol
            })

    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'reaction', 'A', 'b', 'Ea'])
        writer.writeheader()
        writer.writerows(active_reactions)





def main():
    json_file = 'E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/absolute/generation_stats.json'
    best_genome_file = 'E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/absolute/best_genome.json'
    csv_file_removed = 'E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/absolute/removed_reactions.csv'
    csv_file_active = 'E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/absolute/active_reactions.csv'
    reduced_mech_file = 'E:/PPP_WS2024-25/ChemicalMechanismReduction/outputs/absolute/reduced_mech_64_rxns.yaml'  # Path to your reduced mech YAML

    print("Loading GRI30 mechanism reactions...")
    reaction_list = load_reactions_from_gri30()
    print(f"Total reactions loaded: {len(reaction_list)}")

    print(f"Loading generation data from {json_file} ...")
    data = load_generation_data(json_file)

    print(f"Loading best genome from {best_genome_file} ...")
    best_genome = load_best_genome(best_genome_file)

    print("Tracking permanently removed reactions excluding those in best genome...")
    removed_reactions = track_permanently_removed_reactions(data, reaction_list, best_genome)
    print(f"Found {len(removed_reactions)} permanently removed reactions.")
    save_removed_reactions_to_csv(removed_reactions, csv_file_removed)

    print("Saving active reactions with kinetics from reduced mechanism...")
    save_active_reactions(best_genome, reduced_mech_file, csv_file_active)

    print("Done.")

if __name__ == "__main__":
    main()