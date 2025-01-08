import cantera as ct
gas = ct.Solution("gri30.yaml")
reaction = gas.reactions()[0]
print(reaction.to_yaml_string())