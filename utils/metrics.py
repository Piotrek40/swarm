"""Metrics calculation utilities for NanoAI."""

from typing import Dict, List, Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from torch import Tensor, nn, stack, mean
from torch.types import Number


def calculate_metrics(outputs: Tensor, targets: Tensor, problem_type: str) -> Dict[str, float]:
    """
    Calculate various metrics based on the problem type.

    Args:
        outputs: The model's outputs.
        targets: The true target values.
        problem_type: The type of problem ('classification' or 'regression').

    Returns:
        A dictionary containing the calculated metrics.

    Raises:
        ValueError: If an unknown problem type is provided.
    """
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()

    if problem_type == "classification":
        pred = np.argmax(outputs_np, axis=1)
        return {
            "accuracy": accuracy_score(targets_np, pred),
            "f1_score": f1_score(targets_np, pred, average="weighted"),
        }
    elif problem_type == "regression":
        mse = mean_squared_error(targets_np, outputs_np)
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2_score": r2_score(targets_np, outputs_np),
        }
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def calculate_model_complexity(model: nn.Module) -> int:
    """
    Calculate the complexity of a model based on its number of parameters.

    Args:
        model: The model to analyze.

    Returns:
        The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())


def calculate_diversity(population: List[nn.Module]) -> float:
    """
    Calculate the diversity of a population of models.

    Args:
        population: A list of models.

    Returns:
        A measure of population diversity.
    """
    complexities = [calculate_model_complexity(model) for model in population]
    return np.std(complexities) / np.mean(complexities)


def calculate_ensemble_performance(
    ensemble: List[nn.Module], inputs: Tensor, targets: Tensor, problem_type: str
) -> Dict[str, float]:
    """
    Calculate the performance of an ensemble of models.

    Args:
        ensemble: A list of models forming the ensemble.
        inputs: The input data.
        targets: The true target values.
        problem_type: The type of problem ('classification' or 'regression').

    Returns:
        A dictionary containing the ensemble's performance metrics.
    """
    outputs = stack([model(inputs) for model in ensemble])
    ensemble_output = mean(outputs, dim=0)
    return calculate_metrics(ensemble_output, targets, problem_type)


def track_fitness_over_time(swarm: "EvolutionarySwarm", generations: int) -> List[Number]:
    """
    Track the best fitness of a swarm over multiple generations.

    Args:
        swarm: The swarm to track.
        generations: The number of generations to track.

    Returns:
        A list of best fitness values for each generation.
    """
    fitness_history = []
    for _ in range(generations):
        swarm.evolve()
        fitness_history.append(swarm.get_best_model().fitness)
    return fitness_history


def calculate_pareto_front(
    population: List[nn.Module], objectives: List[Callable[[nn.Module], Number]]
) -> List[nn.Module]:
    """
    Calculate the Pareto front for a population based on multiple objectives.

    Args:
        population: A list of models.
        objectives: A list of objective functions to minimize.

    Returns:
        The models on the Pareto front.
    """
    pareto_front = []
    for model in population:
        is_dominated = False
        model_objectives = [obj(model) for obj in objectives]
        for other_model in population:
            if model != other_model:
                other_objectives = [obj(other_model) for obj in objectives]
                if all(o1 <= o2 for o1, o2 in zip(other_objectives, model_objectives)) and any(
                    o1 < o2 for o1, o2 in zip(other_objectives, model_objectives)
                ):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_front.append(model)
    return pareto_front
