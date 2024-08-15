"""Experiment runner for NanoAI."""

import torch
import os
import time
import json
import csv
from typing import List, Dict, Any, Tuple
from models.swarm import EvolutionarySwarm
from utils.logger import Logger
from utils.data_loader import load_data, get_dataset_info
from utils.metrics import (
    calculate_metrics,
    calculate_model_complexity,
    calculate_diversity,
    track_fitness_over_time,
)
from utils.visualizer import Visualizer
from config import *


def run_experiments(experiment_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run experiments based on the provided configurations.

    Args:
        experiment_configs (List[Dict[str, Any]]): List of experiment configurations.

    Returns:
        List[Dict[str, Any]]: Results of all experiments.
    """
    logger = Logger(os.path.join(LOG_DIR, "experiments"))
    all_results = []

    try:
        for config_idx, config in enumerate(experiment_configs):
            dataset_name = config["dataset"]
            logger.log_event(
                f"\nRunning experiment for {dataset_name} dataset with config: {config}"
            )

            dataset_config = get_dataset_info(dataset_name)
            train_loader, val_loader, test_loader = load_data(dataset_name)

            swarm = EvolutionarySwarm(
                dataset_config=dataset_config, initial_population_size=config["population_size"]
            )

            start_time = time.time()

            best_val_loss = float("inf")
            generations_without_improvement = 0

            generations = []
            fitness_values = []
            diversity_values = []

            for generation in range(config["generations"]):
                logger.log_event(f"Generation {generation + 1}/{config['generations']}")

                try:
                    for batch in train_loader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        try:
                            swarm.parallel_evolve(inputs, targets)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                logger.log_warning(
                                    f"CUDA out of memory in generation {generation + 1}. Skipping batch."
                                )
                                if hasattr(torch.cuda, "empty_cache"):
                                    torch.cuda.empty_cache()
                            else:
                                raise

                    val_loss, val_metrics = validate(
                        swarm, val_loader, dataset_config["problem_type"]
                    )

                    diversity = calculate_diversity(
                        [model for subpop in swarm.subpopulations for model in subpop.models]
                    )

                    stats = swarm.get_population_stats()
                    logger.log_event(
                        f"Validation Loss: {val_loss:.4f}, Best Fitness: {stats['best_fitness']:.4f}, Diversity: {diversity:.4f}"
                    )
                    logger.log_event(f"Validation Metrics: {val_metrics}")

                    generations.append(generation)
                    fitness_values.append(stats["best_fitness"])
                    diversity_values.append(diversity)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        generations_without_improvement = 0
                        save_best_model(swarm, dataset_name, config_idx)
                        logger.log_event(f"New best model saved. Validation Loss: {val_loss:.4f}")
                    else:
                        generations_without_improvement += 1

                    if generations_without_improvement >= EARLY_STOPPING_PATIENCE:
                        logger.log_event(
                            f"Early stopping triggered after {generation + 1} generations"
                        )
                        break

                    if (generation + 1) % REPORT_INTERVAL == 0:
                        save_partial_results(
                            os.path.join(
                                LOG_DIR, f"partial_results_{dataset_name}_config_{config_idx}.jsonl"
                            ),
                            generation,
                            val_loss,
                            stats,
                            val_metrics,
                        )

                    if (generation + 1) % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(swarm, dataset_name, config_idx, generation)

                    if generation % 10 == 0:
                        Visualizer.plot_fitness_over_time(
                            generations, fitness_values, dataset_name, config_idx
                        )
                        Visualizer.plot_population_diversity(
                            diversity_values, dataset_name, config_idx
                        )

                except Exception as e:
                    logger.log_error(f"Error in generation {generation + 1}: {str(e)}")
                    continue

            end_time = time.time()

            test_loss, test_metrics = validate(swarm, test_loader, dataset_config["problem_type"])

            final_stats = swarm.get_population_stats()
            result = {
                "dataset": dataset_name,
                "config": config,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "training_time": end_time - start_time,
                "final_stats": final_stats,
                "best_model_complexity": calculate_model_complexity(swarm.get_best_model()),
            }
            all_results.append(result)

            save_final_results(
                os.path.join(RESULT_DIR, f"final_results_{dataset_name}_config_{config_idx}.json"),
                result,
            )

            logger.log_event(f"Test Loss: {test_loss:.4f}")
            logger.log_event(f"Test Metrics: {test_metrics}")
            logger.log_event(f"Training Time: {end_time - start_time:.2f} seconds")

            Visualizer.plot_fitness_over_time(generations, fitness_values, dataset_name, config_idx)
            Visualizer.plot_population_diversity(diversity_values, dataset_name, config_idx)

        save_experiment_summary(os.path.join(RESULT_DIR, "experiment_summary.csv"), all_results)

    except Exception as e:
        logger.log_error(f"An error occurred: {str(e)}")
        import traceback

        logger.log_error(traceback.format_exc())

    finally:
        logger.close()

    return all_results


def validate(
    swarm: EvolutionarySwarm, data_loader: torch.utils.data.DataLoader, problem_type: str
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the swarm's best model on a given data loader.

    Args:
        swarm (EvolutionarySwarm): The swarm to validate.
        data_loader (torch.utils.data.DataLoader): The data loader for validation.
        problem_type (str): The type of problem (classification or regression).

    Returns:
        Tuple[float, Dict[str, float]]: The validation loss and metrics.
    """
    total_loss = 0
    all_outputs = []
    all_targets = []
    swarm.eval()
    best_model = swarm.get_best_model()
    if best_model is None:
        logger.log_warning("Warning: No best model found. Skipping validation.")
        return float("inf"), {}
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = best_model(inputs)
            if problem_type == "regression":
                loss = torch.nn.functional.mse_loss(outputs, targets)
            else:  # classification
                loss = torch.nn.functional.cross_entropy(outputs, targets)
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_targets.append(targets)

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_outputs, all_targets, problem_type)

    return total_loss / len(data_loader), metrics


def save_best_model(swarm: EvolutionarySwarm, dataset_name: str, config_idx: int) -> None:
    """
    Save the best model from the swarm.

    Args:
        swarm (EvolutionarySwarm): The swarm containing the best model.
        dataset_name (str): Name of the dataset.
        config_idx (int): Index of the configuration.
    """
    best_model_path = os.path.join(RESULT_DIR, f"best_model_{dataset_name}_config_{config_idx}.pth")
    torch.save(swarm.get_best_model().state_dict(), best_model_path)


def save_checkpoint(
    swarm: EvolutionarySwarm, dataset_name: str, config_idx: int, generation: int
) -> None:
    """
    Save a checkpoint of the swarm.

    Args:
        swarm (EvolutionarySwarm): The swarm to save.
        dataset_name (str): Name of the dataset.
        config_idx (int): Index of the configuration.
        generation (int): Current generation number.
    """
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"checkpoint_{dataset_name}_config_{config_idx}_gen_{generation}.pth"
    )
    swarm.save_checkpoint(checkpoint_path)


def save_partial_results(
    file_path: str,
    generation: int,
    val_loss: float,
    stats: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    """
    Save partial results during the experiment.

    Args:
        file_path (str): Path to save the partial results.
        generation (int): Current generation number.
        val_loss (float): Validation loss.
        stats (Dict[str, Any]): Statistics of the swarm.
        metrics (Dict[str, float]): Validation metrics.
    """
    try:
        with open(file_path, "a") as f:
            json.dump(
                {"generation": generation, "val_loss": val_loss, "metrics": metrics, **stats}, f
            )
            f.write("\n")
    except Exception as e:
        logger.log_error(f"Error saving partial results: {str(e)}")


def save_final_results(file_path: str, result: Dict[str, Any]) -> None:
    """
    Save final results of the experiment.

    Args:
        file_path (str): Path to save the final results.
        result (Dict[str, Any]): Final results of the experiment.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        logger.log_error(f"Error saving final results: {str(e)}")


def save_experiment_summary(file_path: str, all_results: List[Dict[str, Any]]) -> None:
    """
    Save a summary of all experiments.

    Args:
        file_path (str): Path to save the experiment summary.
        all_results (List[Dict[str, Any]]): Results of all experiments.
    """
    try:
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["dataset", "config", "test_loss", "test_metrics", "training_time"]
            )
            writer.writeheader()
            for result in all_results:
                writer.writerow(
                    {
                        "dataset": result["dataset"],
                        "config": str(result["config"]),
                        "test_loss": result["test_loss"],
                        "test_metrics": str(result["test_metrics"]),
                        "training_time": result["training_time"],
                    }
                )
    except Exception as e:
        logger.log_error(f"Error saving experiment summary: {str(e)}")


def continue_experiment(checkpoint_path: str, config: Dict[str, Any]) -> EvolutionarySwarm:
    """
    Continue an experiment from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        config (Dict[str, Any]): Configuration for the experiment.

    Returns:
        EvolutionarySwarm: The loaded swarm from the checkpoint.
    """
    try:
        swarm = EvolutionarySwarm(
            dataset_config=get_dataset_info(config["dataset"]),
            initial_population_size=config["population_size"],
        )
        swarm.load_checkpoint(checkpoint_path)
        return swarm
    except Exception as e:
        logger.log_error(f"Error continuing experiment from checkpoint: {str(e)}")
        raise


def train_step(
    swarm: EvolutionarySwarm,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    targets: torch.Tensor,
    l2_lambda: float = L2_LAMBDA,
) -> float:
    """
    Perform a single training step.

    Args:
        swarm (EvolutionarySwarm): The swarm to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        data (torch.Tensor): Input data.
        targets (torch.Tensor): Target values.
        l2_lambda (float): L2 regularization coefficient.

    Returns:
        float: The loss value for this training step.
    """
    try:
        optimizer.zero_grad()
        best_model = swarm.get_best_model()
        outputs = best_model(data)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        l2_reg = best_model.get_l2_regularization()
        loss += l2_lambda * l2_reg
        loss.backward()
        optimizer.step()
        return loss.item()
    except Exception as e:
        logger.log_error(f"Error in training step: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        results = run_experiments(EXPERIMENT_CONFIGS)
        print("All experiments completed successfully.")
        print(f"Results saved in {RESULT_DIR}")
    except Exception as e:
        print(f"An error occurred during experiments: {str(e)}")
        import traceback

        traceback.print_exc()
