"""
Initialization module for the experiments package.

This module imports and exposes the main function for running experiments in the NanoAI project.
"""

from typing import List, Dict, Any

from .experiment_runner import run_experiments

__all__ = ['run_experiments']

def get_available_experiments() -> List[str]:
    """
    Returns a list of available experiment configurations.

    This function could be expanded in the future to dynamically detect
    available experiments or configurations.

    Returns:
        A list of strings representing available experiment names or configurations.
    """
    return ["default_experiment"]  # placeholder, expand as needed

def validate_experiment_config(config: Dict[str, Any]) -> bool:
    """
    Validates an experiment configuration.

    Args:
        config: A dictionary containing the experiment configuration.

    Returns:
        True if the configuration is valid, False otherwise.

    Raises:
        ValueError: If the configuration is invalid.
    """
    required_keys = ["dataset", "population_size", "generations"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in experiment configuration: {key}")
    return True
