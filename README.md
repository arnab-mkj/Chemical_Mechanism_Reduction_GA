# Genetic Algorithm for Chemical Mechanism Optimization

## Overview

This project implements a genetic algorithm to optimize and reduce chemical reaction mechanisms using Cantera. The goal is to create reduced mechanisms that maintain accuracy while minimizing the number of reactions and species, improving computational efficiency for simulations.

## Features

### Current Features

- Genetic algorithm implementation for mechanism reduction
- Fitness evaluation based on:
  - Temperature profiles
  - Species concentrations
  - Ignition delay times
- Real-time plotting of:
  - Species evolution
  - Fitness scores
  - Temperature profiles
- Mechanism validation against full mechanism
- Results export in YAML/JSON formats

### Features in Progress

- **Elitism**: Preserving best genomes across generations
- **Species Sensitivity Analysis**: Automating sensitivity-based reduction
- **Real-Time Plotting**: Adding animations for:
  - Fitness evolution
  - Species profiles
  - Population diversity
- **GUI Dashboard**: User interface for:
  - Running simulations
  - Parameter selection
  - Results visualization
- **Multi-Objective Optimization**: Balancing:
  - Mechanism accuracy
  - Size reduction
  - Computational cost

## Installation

### Prerequisites

- Python 3.8+
- Cantera 3.0+
- Git

### Setup Steps

1. **Clone the Repository**

```
git clone https://github.com/yourusername/chemical-mechanism-optimization.gitcd chemical-mechanism-optimization
```

1. **Create Virtual Environment**

```
python -m venv venvsource venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install Dependencies**

```
pip install -r requirements.txt
```

1. **Install Cantera**

- Follow instructions at [Cantera Installation Guide](https://cantera.org/install/index.html)

## Project Structure

```
chemical-mechanism-optimization/├── main.py                 # Main script to run the genetic algorithm├── genetic_algorithm.py    # Implementation of the genetic algorithm├── fitness_evaluator.py    # Fitness function for mechanism evaluation├── simulation_runner.py    # Handles Cantera simulations├── sensitivity_analysis.py # Species sensitivity analysis├── requirements.txt        # Python dependencies├── output/                 # Directory for saving results└── README.md               # Project documentation
```

## Usage

### Running the Genetic Algorithm

```
python main.py
```

### Configuration

Edit `config.yaml` to set:

- Population size
- Number of generations
- Mutation rate
- Crossover probability
- Target species
- Reactor conditions

### Output Files

- Reduced mechanisms: `output/mechanisms/`
- Species concentrations: `output/species/`
- Fitness scores: `output/fitness/`
- Plots: `output/plots/`

## Future Work

### Short Term

- Implement elitism
- Add species sensitivity analysis
- Improve real-time plotting
- Create GUI dashboard

### Long Term

- **Parallelization**
  - Implement multiprocessing
  - Distribute workload across cores

- **Advanced Visualization**
  - Interactive plots with Plotly
  - 3D visualizations
  - Animation of mechanism evolution
- **Benchmarking**
  - Compare against experimental data
  - Validate with different mechanisms
  - Performance metrics


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- Cantera Development Team
- Research Group/Supervisor
- Contributors

---

_Last Updated: [08.01.2025]_