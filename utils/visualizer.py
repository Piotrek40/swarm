"""Visualization utilities for NanoAI experiments."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple
import os
import torch
from config import RESULT_DIR


class Visualizer:
    """Class for creating visualizations of experiment results."""

    @staticmethod
    def plot_fitness_over_time(
        generations: List[int], fitness_values: List[float], dataset_name: str, config_idx: int
    ) -> None:
        """
        Plots the fitness values over generations.

        Args:
            generations (List[int]): List of generation numbers.
            fitness_values (List[float]): Corresponding fitness values.
            dataset_name (str): Name of the dataset.
            config_idx (int): Index of the configuration.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(generations, fitness_values)
        plt.title(f"Fitness over Generations - {dataset_name} (Config {config_idx})")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.savefig(
            os.path.join(RESULT_DIR, f"fitness_plot_{dataset_name}_config_{config_idx}.png")
        )
        plt.close()

    @staticmethod
    def plot_population_diversity(
        diversity_values: List[float], dataset_name: str, config_idx: int
    ) -> None:
        """
        Plots the population diversity over generations.

        Args:
            diversity_values (List[float]): List of diversity values.
            dataset_name (str): Name of the dataset.
            config_idx (int): Index of the configuration.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(diversity_values)), diversity_values)
        plt.title(f"Population Diversity - {dataset_name} (Config {config_idx})")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.savefig(
            os.path.join(RESULT_DIR, f"diversity_plot_{dataset_name}_config_{config_idx}.png")
        )
        plt.close()

    @staticmethod
    def plot_pareto_front(
        objective1_values: List[float],
        objective2_values: List[float],
        dataset_name: str,
        config_idx: int,
    ) -> None:
        """
        Plots the Pareto front for two objectives.

        Args:
            objective1_values (List[float]): Values for the first objective.
            objective2_values (List[float]): Values for the second objective.
            dataset_name (str): Name of the dataset.
            config_idx (int): Index of the configuration.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(objective1_values, objective2_values)
        plt.title(f"Pareto Front - {dataset_name} (Config {config_idx})")
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.savefig(
            os.path.join(RESULT_DIR, f"pareto_front_{dataset_name}_config_{config_idx}.png")
        )
        plt.close()

    @staticmethod
    def plot_model_complexity_distribution(
        complexities: List[int], dataset_name: str, config_idx: int
    ) -> None:
        """
        Plots the distribution of model complexities in the population.

        Args:
            complexities (List[int]): List of model complexities.
            dataset_name (str): Name of the dataset.
            config_idx (int): Index of the configuration.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(complexities, kde=True)
        plt.title(f"Model Complexity Distribution - {dataset_name} (Config {config_idx})")
        plt.xlabel("Complexity (Number of Parameters)")
        plt.ylabel("Frequency")
        plt.savefig(
            os.path.join(
                RESULT_DIR, f"complexity_distribution_{dataset_name}_config_{config_idx}.png"
            )
        )
        plt.close()

    @staticmethod
    def plot_learning_curves(
        train_losses: List[float], val_losses: List[float], dataset_name: str, config_idx: int
    ) -> None:
        """
        Plots the learning curves (train and validation losses).

        Args:
            train_losses (List[float]): List of training losses.
            val_losses (List[float]): List of validation losses.
            dataset_name (str): Name of the dataset.
            config_idx (int): Index of the configuration.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
        plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
        plt.title(f"Learning Curves - {dataset_name} (Config {config_idx})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(RESULT_DIR, f"learning_curves_{dataset_name}_config_{config_idx}.png")
        )
        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray, class_names: List[str], dataset_name: str, config_idx: int
    ) -> None:
        """
        Plots a confusion matrix.

        Args:
            cm (np.ndarray): The confusion matrix.
            class_names (List[str]): List of class names.
            dataset_name (str): Name of the dataset.
            config_idx (int): Index of the configuration.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
        )
        plt.title(f"Confusion Matrix - {dataset_name} (Config {config_idx})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(
            os.path.join(RESULT_DIR, f"confusion_matrix_{dataset_name}_config_{config_idx}.png")
        )
        plt.close()

    @staticmethod
    def plot_feature_importance(
        feature_importance: List[float],
        feature_names: List[str],
        dataset_name: str,
        config_idx: int,
    ) -> None:
        """
        Plots feature importance.

        Args:
            feature_importance (List[float]): List of feature importance scores.
            feature_names (List[str]): List of feature names.
            dataset_name (str): Name of the dataset.
            config_idx (int): Index of the configuration.
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(x=feature_importance, y=feature_names)
        plt.title(f"Feature Importance - {dataset_name} (Config {config_idx})")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.savefig(
            os.path.join(RESULT_DIR, f"feature_importance_{dataset_name}_config_{config_idx}.png")
        )
        plt.close()

    @staticmethod
    def plot_model_architecture(model: torch.nn.Module, dataset_name: str, config_idx: int) -> None:
        """
        Plots the architecture of a model.

        Args:
            model (torch.nn.Module): The model to visualize.
            dataset_name (str): Name of the dataset.
            config_idx (int): Index of the configuration.
        """
        from torchviz import make_dot

        x = torch.randn(1, model.input_size).requires_grad_(True)
        y = model(x)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.render(
            os.path.join(RESULT_DIR, f"model_architecture_{dataset_name}_config_{config_idx}"),
            format="png",
        )
