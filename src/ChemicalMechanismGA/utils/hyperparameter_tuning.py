import itertools
import random
from src.ChemicalMechanismGA.components.fitness_function import FitnessEvaluator

class HyperparameterTuner:
    def __init__(self, fitness_evaluator_class, results):
        """
        Initialize the HyperparameterTuner.

        Parameters:
            fitness_evaluator_class (class): The FitnessEvaluator class to evaluate fitness.
            results (dict): Simulation results to evaluate fitness.
        """
        self.fitness_evaluator_class = fitness_evaluator_class
        self.results = results

    def grid_search(self, param_grid):
        """
        Perform grid search to find the best hyperparameters.

        Parameters:
            param_grid (dict): Dictionary of hyperparameters and their possible values.

        Returns:
            dict: Best hyperparameters and their corresponding fitness score.
        """
        # Generate all combinations of hyperparameters
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        best_params = None
        best_fitness = float("inf")

        # Evaluate each combination of hyperparameters
        for params in param_combinations:
            print(f"Testing hyperparameters: {params}")
            evaluator = self.fitness_evaluator_class(
                target_temperature=2000.0,
                target_species={"CO2": 0.1, "H2O": 0.2},
                target_delay=1.0,
                weight_temperature=params["weight_temperature"],
                weight_species=params["weight_species"],
                weight_ignition_delay=params["weight_ignition_delay"],
                difference_function=params["difference_function"]
            )
            fitness = evaluator.combined_fitness(self.results)
            print(f"Fitness Score: {fitness}")

            # Update the best parameters if this combination is better
            if fitness < best_fitness:
                best_fitness = fitness
                best_params = params

        return {"best_params": best_params, "best_fitness": best_fitness}

    def random_search(self, param_ranges, n_iter=10):
        """
        Perform random search to find the best hyperparameters.

        Parameters:
            param_ranges (dict): Dictionary of hyperparameters and their ranges.
            n_iter (int): Number of random combinations to evaluate.

        Returns:
            dict: Best hyperparameters and their corresponding fitness score.
        """
        best_params = None
        best_fitness = float("inf")

        for _ in range(n_iter):
            # Randomly sample hyperparameters
            params = {
                "weight_temperature": random.uniform(*param_ranges["weight_temperature"]),
                "weight_species": random.uniform(*param_ranges["weight_species"]),
                "weight_ignition_delay": random.uniform(*param_ranges["weight_ignition_delay"]),
                "difference_function": random.choice(param_ranges["difference_function"])
            }
            print(f"Testing hyperparameters: {params}")
            evaluator = self.fitness_evaluator_class(
                target_temperature=2000.0,
                target_species={"CO2": 0.1, "H2O": 0.2},
                target_delay=1.0,
                weight_temperature=params["weight_temperature"],
                weight_species=params["weight_species"],
                weight_ignition_delay=params["weight_ignition_delay"],
                difference_function=params["difference_function"]
            )
            fitness = evaluator.combined_fitness(self.results)
            print(f"Fitness Score: {fitness}")

            # Update the best parameters if this combination is better
            if fitness < best_fitness:
                best_fitness = fitness
                best_params = params

        return {"best_params": best_params, "best_fitness": best_fitness}