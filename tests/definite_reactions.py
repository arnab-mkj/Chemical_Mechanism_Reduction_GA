import cantera as ct

# List of definite species
species_def = ['CH4', 'O2', 'N2', 'Ar', 'CO2', 'H2O', 'CO', 'H2', 'O', 'OH', 'H', 'CH3']

def def_reactions_with_third_body():
    # Load the mechanism
    gas = ct.Solution('gri30.yaml')
    reactions = gas.reactions()  # Get all reactions in the mechanism
    definite_reaction_indices = []  # List to store indices of definite reactions

    # Iterate through all reactions
    for i, reaction in enumerate(reactions):
        # Get the reactants and products of the reaction
        reactants = reaction.reactants.keys()
        products = reaction.products.keys()

        # Check if the reaction is a third-body reaction
        if isinstance(reaction, ct.ThreeBodyReaction):
            # Get third-body efficiencies
            third_body_species = reaction.efficiencies.keys()
            # Check if all reactants, products, and third-body species are in the definite species list
            if (all(species in species_def for species in reactants) and
                all(species in species_def for species in products) and
                all(species in species_def for species in third_body_species)):
                definite_reaction_indices.append(i)  # Add the index of the reaction
        else:
            # For non-third-body reactions, check only reactants and products
            if all(species in species_def for species in reactants) and all(species in species_def for species in products):
                definite_reaction_indices.append(i)  # Add the index of the reaction

    return definite_reaction_indices

# Example usage
definite_reaction_indices = def_reactions_with_third_body()
print("Indices of definite reactions (including third-body reactions):", definite_reaction_indices)