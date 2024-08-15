"""
Utility modules for NanoAI project.

This module imports and exposes various utility functions and classes used throughout the NanoAI project.
"""

from typing import Dict, List, Tuple, Any

from torch import Tensor, nn

from .data_loader import (
    load_data,
    get_dataset_info,
    get_problem_type,
    get_input_output_sizes,
)
from .logger import Logger
from .metrics import (
    calculate_metrics,
    calculate_model_complexity,
    calculate_diversity,
    calculate_ensemble_performance,
    track_fitness_over_time,
    calculate_pareto_front,
)
from .visualizer import Visualizer

__all__ = [
    "load_data",
    "get_dataset_info",
    "get_problem_type",
    "get_input_output_sizes",
    "Logger",
    "calculate_metrics",
    "calculate_model_complexity",
    "calculate_diversity",
    "calculate_ensemble_performance",
    "track_fitness_over_time",
    "calculate_pareto_front",
    "Visualizer",
]


def load_and_prepare_data(dataset_name: str) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Load and prepare data for a given dataset.

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        A tuple containing train, validation, and test data loaders.

    Raises:
        ValueError: If the dataset is not supported.
    """
    try:
        return load_data(dataset_name)
    except ValueError as e:
        raise ValueError(f"Error loading dataset {dataset_name}: {str(e)}") from e


def get_model_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get model information for a given dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        A dictionary containing model information.

    Raises:
        ValueError: If the dataset is not supported.
    """
    try:
        dataset_info = get_dataset_info(dataset_name)
        problem_type = get_problem_type(dataset_name)
        input_size, output_size = get_input_output_sizes(dataset_name)
        return {
            "dataset_info": dataset_info,
            "problem_type": problem_type,
            "input_size": input_size,
            "output_size": output_size,
        }
    except ValueError as e:
        raise ValueError(f"Error getting model info for dataset {dataset_name}: {str(e)}") from e


def evaluate_model(
    model: nn.Module, data: Tensor, targets: Tensor, problem_type: str
) -> Dict[str, float]:
    """
    Evaluate a model on given data.

    Args:
        model: The model to evaluate.
        data: Input data.
        targets: Target values.
        problem_type: Type of problem (e.g., "classification", "regression").

    Returns:
        A dictionary containing evaluation metrics.
    """
    outputs = model(data)
    return calculate_metrics(outputs, targets, problem_type)


# Add any additional utility functions here
