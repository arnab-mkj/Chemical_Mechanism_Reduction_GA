import os
from pathlib import Path
import logging

#logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'ChemicalMechanismGA'

list_of_files = [
    # Source code structure
    f"src/{project_name}/__init__.py",

    # Core components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/mechanism_analyzer.py",
    f"src/{project_name}/components/genetic_algorithm.py",
    f"src/{project_name}/components/fitness_function.py",
    f"src/{project_name}/components/population.py",
    f"src/{project_name}/components/individual.py",

    # GA Operators
    f"src/{project_name}/operators/__init__.py",
    f"src/{project_name}/operators/selection.py",
    f"src/{project_name}/operators/crossover.py",
    f"src/{project_name}/operators/mutation.py",

    # Utility functions
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/yaml_handler.py",
    f"src/{project_name}/utils/mechanism_utils.py",
    f"src/{project_name}/utils/visualization.py",
    f"src/{project_name}/utils/validation.py",

    # Configuration
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    # Constants and logging
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/constants/ga_parameters.py",
    f"src/{project_name}/logger/__init__.py",

    # Configuration files
    "config/config.yaml",
    "params.yaml",

    # Project setup files
    "requirements.txt",
    "setup.py",
    "README.md",

    # Research and notebooks
    "research/trials.ipynb",
    "research/mechanism_analysis.ipynb",
    "research/ga_performance_analysis.ipynb",

    # Data directories
    "data/mechanisms/.gitkeep",     # Original mechanism files
    "data/reduced/.gitkeep",        # Reduced mechanism files
    "data/results/.gitkeep",        # GA results and statistics

    # Tests
    "tests/__init__.py",
    "tests/test_mechanism_analyzer.py",
    "tests/test_genetic_algorithm.py",
    "tests/test_fitness_function.py",
    "tests/test_operators.py",

    # Documentation
    "docs/README.md",
    "docs/ga_framework.md",
    "docs/mechanism_format.md",
    "docs/api_reference.md",

    # GitHub workflows
    ".github/workflows/.gitkeep",
    ".gitignore"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")