import cantera as ct

class GRI30Analyzer:
    def __init__(self, file_path):
        """Initialize the GRI30Analyzer with the path to the GRI30 YAML file."""
        self.file_path = file_path
        self.gas = self.load_gri30_mechanism()

    def load_gri30_mechanism(self):
        """Load the GRI30 mechanism from the YAML file."""
        try:
            gas = ct.Solution(self.file_path)
            return gas
        except Exception as e:
            print(f"Error loading GRI30 mechanism: {e}")
            return None

    def output_species_info(self):
        """Output information about the species in the mechanism."""
        if self.gas is None:
            print("No gas object available. Please check the file path.")
            return

        print("Species Information:")
        for species in self.gas.species():
            print(f"Name: {species.name}")
            print(f"Composition: {species.composition}")
            print(f"Thermo: {species.thermo}")
            print("")

    def output_reaction_info(self):
        """Output information about the reactions in the mechanism."""
        if self.gas is None:
            print("No gas object available. Please check the file path.")
            return

        print("Reaction Information:")
        for reaction in self.gas.reactions():
            print(f"Reaction: {reaction.equation}")
            print(f"Rate Coefficient: {reaction.rate}")
            print("")

    def analyze(self):
        """Run the analysis to output species and reaction information."""
        self.output_species_info()
        self.output_reaction_info()

def main():
 
    gri30_file_path = 'data/output/best_reduced_mechanism.yaml'  # Change this to your actual file path
    analyzer = GRI30Analyzer(gri30_file_path)

    analyzer.analyze()

if __name__ == "__main__":
    main()