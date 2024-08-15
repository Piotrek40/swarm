"""
Main module for running NanoAI experiments.

This module contains the main functionality for setting up and running
experiments, as well as displaying and visualizing results.
"""

import sys
import os
import argparse
import random
from typing import List, Dict, Any

import numpy as np
import torch
from torch import Tensor

from experiments.experiment_runner import run_experiments
from utils.visualizer import Visualizer
from config import EXPERIMENT_CONFIGS, IS_KAGGLE, IS_COLAB, DEVICE, LOG_DIR, RESULT_DIR
from utils.logger import Logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args: argparse.Namespace) -> None:
    """
    Main function to run experiments.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    logger = Logger(os.path.join(LOG_DIR, "main_log"))

    try:
        set_seed(42)

        logger.log_event(f"PyTorch version: {torch.__version__}")
        logger.log_event(f"Using device: {DEVICE}")

        if torch.cuda.is_available():
            logger.log_event(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.log_event(f"CUDA version: {torch.version.cuda}")

        if IS_KAGGLE:
            logger.log_event("Running in Kaggle environment")
        elif IS_COLAB:
            logger.log_event("Running in Google Colab environment")
        else:
            logger.log_event("Running in local environment")

        configs = (
            [config for config in EXPERIMENT_CONFIGS if config["dataset"] == args.dataset]
            if args.dataset
            else EXPERIMENT_CONFIGS
        )

        results = run_experiments(configs)

        if results:
            display_results(results)
            visualize_results(results)
        else:
            logger.log_event("No results returned from experiments")

    except RuntimeError as runtime_error:
        logger.log_error(f"A runtime error occurred during experiments: {str(runtime_error)}")
    except ValueError as value_error:
        logger.log_error(f"A value error occurred during experiments: {str(value_error)}")
    except Exception as exception:
        logger.log_error(f"An unexpected error occurred during experiments: {str(exception)}")
        logger.log_error(traceback.format_exc())
    finally:
        logger.close()


def display_results(results: List[Dict[str, Any]]) -> None:
    """
    Display the results of the experiments.

    Args:
        results (List[Dict[str, Any]]): List of experiment results.
    """
    print("\nExperiment Results:")
    for result in results:
        print(f"\nDataset: {result['dataset']}")
        print(f"Config: {result['config']}")
        print(f"Test Loss: {result['test_loss']:.4f}")
        print(f"Test Metrics: {result['test_metrics']}")
        print(f"Best Model Complexity: {result['best_model_complexity']}")
        if "final_stats" in result:
            print("Final Population Stats:")
            for key, value in result["final_stats"].items():
                print(f"  {key}: {value}")


def visualize_results(results: List[Dict[str, Any]]) -> None:
    """
    Visualize the results of the experiments.

    Args:
        results (List[Dict[str, Any]]): List of experiment results.
    """
    for result in results:
        dataset_name = result["dataset"]
        config_idx = EXPERIMENT_CONFIGS.index(result["config"])

        Visualizer.plot_model_complexity_distribution(
            [result["best_model_complexity"]], dataset_name, config_idx
        )

        if result["test_metrics"].get("confusion_matrix") is not None:
            Visualizer.plot_confusion_matrix(
                result["test_metrics"]["confusion_matrix"],
                list(range(result["config"]["output_size"])),
                dataset_name,
                config_idx,
            )

        if "feature_importance" in result:
            Visualizer.plot_feature_importance(
                result["feature_importance"],
                [f"Feature {i}" for i in range(len(result["feature_importance"]))],
                dataset_name,
                config_idx,
            )

    print(f"Visualizations saved in {RESULT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NanoAI experiments")
    parser.add_argument("--dataset", type=str, help="Specify a dataset to run experiments on")
    args = parser.parse_args()
    main(args)
