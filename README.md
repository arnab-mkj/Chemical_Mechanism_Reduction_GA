# Genetic Algorithm-based Tool for Chemical Mechanism Reduction

 A Python-based tool for the automatic reduction and optimization of chemical reaction mechanisms using genetic algorithms. This software enables efficient simplification of complex chemical kinetics while maintaining accuracy for specific operating conditions.

# Table of Contents

- Overview
- Features
- Installation
- Usage
  - Configuration Tab
  - Tests Tab
  - Reactor Tabs
    - Flame Simulation Tab
    - Constant Pressure Simulation Tab
- Method Description
- Common Workflows
- Troubleshooting
- Additional Tips
- References

# Overview

 This tool implements a genetic algorithm (GA) approach to reduce the size of chemical reaction mechanisms while preserving their ability to describe the overall chemistry with acceptable accuracy. The method considers not only chemical accuracy and mechanism size but also computational cost, helping to avoid stiff and slow-converging mechanisms.

 The reduction process gradually eliminates reactions from a detailed mechanism according to user-defined criteria. The tool also provides optimization capabilities to adjust reaction rate coefficients, improving the accuracy of reduced mechanisms for specific applications.

# Features

- **User-friendly GUI** built with Tkinter for easy configuration and execution
- **Genetic algorithm-based reduction** that automatically identifies and removes unimportant reactions
- **Mechanism optimization** to adjust reaction rate coefficients for improved accuracy
- **Customizable objective functions** to balance accuracy, size, and computational cost
- **Multiple evaluation criteria** including ignition delay time, temperature profiles, and species concentrations
- **Reactor simulations** for mechanism validation (homogeneous reactors and laminar flames)
- **Testing capabilities** for individual GA components (mutation, crossover, fitness evaluation, etc.)
- **Visualization tools** for comparing reduced and detailed mechanisms

# Installation

# Prerequisites

- Python 3.8 or later
- Git (for cloning the repository)

# Setup Steps

1. **Clone the repository**:
 ```bash
    git clone https://github.com/arnab-mkj/Chemical_Mechanism_Reduction_GA.git\
   cd Chemical_Mechanism_Reduction_GA
 ```
2. **Create a virtual environment**:
 ```bash
   python -m venv ./venv
 ```
3. **Activate the virtual environment**:

   - Windows:
     ```bash
      .\venv\Scripts\activate
      ```
    - Unix/macOS:
    ```bash
      source venv/bin/activate
    ```
4. **Install dependencies**:
  ```bash
   pip install --upgrade pip\ -->
   pip install -r requirements.txt
  ```
5. **Launch the application**:
  ```bash
   python app.py
   ```
# Usage

# The GUI is divided into several tabs for different functionalities:

# Configuration Tab

# This tab allows you to set up the genetic algorithm parameters for mechanism reduction:

- **Population Size**: Number of candidate solutions per generation (default: 50)
- **Original Mechanism Path**: Path to the detailed mechanism file (e.g., GRI-Mech 3.0)
- **Output Directory**: Where to save the reduced mechanism and results
- **Crossover Rate**: Probability of crossover between chromosomes (default: 0.4)
- **Mutation Rate**: Probability of mutation for each gene (default: 0.003)
- **Enable Elitism**: Option to preserve the best solutions between generations
- **Elite Size**: Number of best solutions to preserve (if elitism is enabled)
- **Initialize with Reduced Mechanism**: Option to start with a partially reduced mechanism
- **Reduction Threshold**: Fraction for initial reduction (if enabled)
- **Species Weights Method**: Choose between manual input or sensitivity analysis
- **Conditions**: JSON-formatted operating conditions for mechanism evaluation
- **Run Genetic Algorithm**: Button to start the reduction process

# Tests Tab

# This tab allows you to validate individual components of the genetic algorithm:

- **Select Test to Run**: Dropdown menu with options like mutation, fitness, crossover, selection, initialization
- **Test Parameters**: Fields specific to the selected test
- **Run Test**: Button to execute the selected test

# Reactor Tabs

We will need a existing reduced mechanism to run compare with the full mechanism for this. It is recommended to run the reduction process under user-specified conditions to get a reduced mehanism and then come to the below tests.

However, a reduced mech with 64 reactionand 29 species have been added for easy working.

# Flame Simulation Tab

# Simulate and compare reduced vs. full mechanisms in a laminar flame:

- **Reactor Type**: Choose between Laminar Flame Reactor or Constant Pressure Reactor
- **Equivalence Ratio (phi)**: Fuel-oxidizer mixture ratio (default: 2.0)
- **Temperature (K)**: Initial temperature (default: 1000)
- **Pressure (Pa)**: Initial pressure (default: 101325, or 1 atm)
- **Fuel**: Fuel species (default: CH4)
- **Oxidizer**: Oxidizer composition (default: O2:0.21, N2:0.79)
- **Mass Flow Rate**: For Laminar Flame Reactor (default: 0.04 kg/m²/s)
- **Reduced Mechanism File**: Path to the reduced mechanism
- **Full Mechanism Name**: Name of the full mechanism (default: gri30.yaml)

# Results include burning velocities, maximum temperatures, and plots of temperature profiles, species mole fractions, and heat release rates.

# Constant Pressure Simulation Tab

# Simulate and compare mechanisms in a Constant Pressure Reactor:

- **Temperature (K)**: Initial temperature (default: 2561)
- **Pressure (Pa)**: Initial pressure (default: 101325)
- **Equivalence Ratios**: JSON list of ratios to simulate (default: \[0.4\])
- **Fuel**: Fuel species (default: CH4)
- **Oxidizer**: Oxidizer composition (default: O2:0.21, N2:0.79)
- **End Time (s)**: Simulation duration (default: 0.1)
- **Key Species**: JSON list of species to track (default: \["CH4", "O2", "CO2", "CO", "OH"\])
- **Reduced Mechanism File**: Path to the reduced mechanism
- **Full Mechanism Name**: Name of the full mechanism (default: gri30.yaml)
- **Output Directory**: Where to save simulation plots
- **X-Axis Limit (ms)**: Maximum time to display on plots (default: 0.1)

# Results include ignition delay times, maximum temperatures, and plots of temperature and species evolution for each equivalence ratio.

# Method Description

# The genetic algorithm approach for mechanism reduction works as follows:

1. **Initialization**: Create an initial population of chromosomes, each representing a subset of reactions from the detailed mechanism.
2. **Evaluation**: Assess each chromosome's performance by simulating a reactor or flame with the corresponding reduced mechanism and comparing results to the detailed mechanism.
3. **Selection**: Choose chromosomes for reproduction based on their fitness (accuracy, size, computational cost).
4. **Crossover**: Exchange information between selected chromosomes to create new candidate solutions.
5. **Mutation**: Randomly modify a small number of reactions to maintain diversity and explore new possibilities.
6. **Repeat**: Continue the process for multiple generations until a satisfactory reduced mechanism is found.

# The objective function balances multiple criteria:

- Accuracy in predicting ignition delay time
- Accuracy in predicting steady-state temperature
- Number of reactions in the reduced mechanism
- Computational time required for solution

# Common Workflows

# Basic Mechanism Reduction

1. Open the **Configuration** tab
2. Set the path to your detailed mechanism
3. Specify the output directory
4. Keep default values for other parameters
5. Click **Run Genetic Algorithm**
6. Monitor progress in the output text box
7. Review the reduced mechanism in the output directory

# Testing GA Components

1. Switch to the **Tests** tab
2. Select a test type (e.g., "fitness")
3. Set appropriate parameters
4. Click **Run Test**
5. Check results in the output text box

# Validating Reduced Mechanisms

1. Go to the **Flame Simulation** or **Constant Pressure Simulation** tab
2. Set the path to your reduced mechanism
3. Configure simulation parameters
4. Run the simulation
5. Compare results between reduced and full mechanisms

# Troubleshooting

- **GUI doesn't open**: Verify Python and Tkinter installation with `python --version` and `python -m tkinter`
- **Error on run**: Check JSON entries for syntax errors (use an online validator if needed)
- **No results**: Ensure mechanism files exist and output directory is writable
- **Test fails**: Verify test files are in the correct location (src/ChemicalMechanismGA/tests/)
- **Slow convergence**: Try adjusting GA parameters (population size, mutation rate, etc.)

# Additional Tips

- **Scroll when needed**: Use the scroll bar to see all available options
- **Check error messages**: Read carefully to identify and fix issues
- **Experiment safely**: Start with defaults, then change one parameter at a time
- **Use the output text**: It provides valuable information about the process and results
- **Terminal output**: Some print statements appear in the terminal rather than the GUI content here. You can paste directly from Word or other rich text sources.

# References

- GRI-Mech 3.0 Documentation
- Cantera Documentation
- Genetic Algorithms in Optimization