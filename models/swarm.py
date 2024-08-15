import torch
import random
import numpy as np
import gc
import concurrent.futures
from typing import List, Dict, Any, Tuple
from models.nano_model import NanoModel, SymbioticPair
from config import *


class Environment:
    """
    Represents the environment in which the models evolve.
    """

    def __init__(self):
        self.current_condition = "normal"

    def change(self) -> None:
        """Changes the current environmental condition randomly."""
        self.current_condition = random.choice(ENVIRONMENT_CONDITIONS)


class Niche:
    """
    Represents a specialized niche for model evaluation.

    Args:
        specialization (str): The specialization of the niche.
    """

    def __init__(self, specialization: str):
        self.specialization = specialization

    def evaluate(self, model: NanoModel, environment: Environment, data: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Evaluates a model in the current niche and environment.

        Args:
            model (NanoModel or SymbioticPair): The model to evaluate.
            environment (Environment): The current environment.
            data (torch.Tensor): The input data.
            targets (torch.Tensor): The target values.

        Returns:
            float: The negative loss (fitness) of the model.
        """
        try:
            with torch.no_grad():
                outputs = model(data)
                if outputs.shape != targets.shape:
                    outputs = outputs.view(targets.shape)
                loss = torch.nn.functional.mse_loss(outputs, targets)

            if (
                self.specialization == "outlier_focus"
                and environment.current_condition == "challenging"
            ):
                residuals = (outputs - targets).abs()
                loss = (residuals**2).mean() + residuals.max()
            elif (
                self.specialization == "noise_resistant"
                and environment.current_condition == "extreme"
            ):
                noisy_data = data + torch.randn_like(data) * 0.1
                noisy_outputs = model(noisy_data)
                loss = torch.nn.functional.mse_loss(noisy_outputs, targets)

            return -loss.item()
        except Exception as e:
            print(f"Error in niche evaluation: {str(e)}")
            return float("-inf")


class Subpopulation:
    """
    Represents a subpopulation of models.

    Args:
        models (List[NanoModel]): A list of NanoModel or SymbioticPair instances.
        niche (Niche): The niche for this subpopulation.
    """

    def __init__(self, models: List[NanoModel], niche: Niche):
        self.models = models
        self.niche = niche
        self.mutation_rate = INITIAL_MUTATION_RATE
        self.crossover_rate = INITIAL_CROSSOVER_RATE

    def adapt(self, environment: Environment, data: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Adapts the subpopulation to the current environment.

        Args:
            environment (Environment): The current environment.
            data (torch.Tensor): The input data.
            targets (torch.Tensor): The target values.
        """
        try:
            for model in self.models:
                model.fitness = self.niche.evaluate(model, environment, data, targets)
                if random.random() < GENETIC_DRIFT_PROBABILITY:
                    model.random_modification()
            self.selection()
            self.reproduce()
            if ADAPTIVE_MUTATION_RATE:
                self.adapt_mutation_rate()
            if ADAPTIVE_CROSSOVER_RATE:
                self.adapt_crossover_rate()
        except Exception as e:
            print(f"Error in subpopulation adaptation: {str(e)}")

    def selection(self) -> None:
        """Performs selection on the subpopulation."""
        self.models.sort(key=lambda m: m.fitness, reverse=True)
        self.models = self.models[: len(self.models) // 2]

    def reproduce(self) -> None:
        """Reproduces the subpopulation through crossover and mutation."""
        new_models = []
        while len(new_models) + len(self.models) < len(self.models) * 2:
            if random.random() < SYMBIOSIS_PROBABILITY:
                parent1, parent2 = random.sample(self.models, 2)
                child = SymbioticPair(parent1.clone(), parent2.clone())
            else:
                parent1, parent2 = random.sample(self.models, 2)
                child = self.crossover(parent1, parent2)
            child.mutate(self.mutation_rate)
            new_models.append(child)
        self.models.extend(new_models)

    def crossover(self, parent1: NanoModel, parent2: NanoModel) -> NanoModel:
        """
        Performs crossover between two parent models.

        Args:
            parent1 (NanoModel or SymbioticPair): The first parent.
            parent2 (NanoModel or SymbioticPair): The second parent.

        Returns:
            NanoModel or SymbioticPair: The child model.
        """
        child = parent1.clone()
        for child_param, parent2_param in zip(child.parameters(), parent2.parameters()):
            mask = (torch.rand_like(child_param) < self.crossover_rate).to(child_param.device)
            child_param.data[mask] = parent2_param.data[mask]
        return child

    def adapt_mutation_rate(self) -> None:
        """Adapts the mutation rate based on population fitness."""
        fitness_improvement = (self.models[0].fitness - self.models[-1].fitness) / abs(
            self.models[-1].fitness
        )
        if fitness_improvement > 0.1:
            self.mutation_rate = max(self.mutation_rate * 0.9, MUTATION_RATE_RANGE[0])
        else:
            self.mutation_rate = min(self.mutation_rate * 1.1, MUTATION_RATE_RANGE[1])

    def adapt_crossover_rate(self) -> None:
        """Adapts the crossover rate based on population fitness."""
        avg_fitness = sum(m.fitness for m in self.models) / len(self.models)
        if avg_fitness > self.models[0].fitness * 0.9:
            self.crossover_rate = min(self.crossover_rate * 1.1, CROSSOVER_RATE_RANGE[1])
        else:
            self.crossover_rate = max(self.crossover_rate * 0.9, CROSSOVER_RATE_RANGE[0])


class EvolutionarySwarm:
    """
    Represents the main evolutionary swarm of models.

    Args:
        dataset_config (Dict[str, Any]): The configuration for the dataset.
        initial_population_size (int): The initial size of the population.
    """

    def __init__(self, dataset_config: Dict[str, Any], initial_population_size: int):
        if not isinstance(dataset_config, dict):
            raise TypeError("dataset_config must be a dictionary")
        if not isinstance(initial_population_size, int) or initial_population_size <= 0:
            raise ValueError("initial_population_size must be a positive integer")

        self.dataset_config = dataset_config
        self.subpopulations = [
            Subpopulation(
                [
                    NanoModel(dataset_config)
                    for _ in range(initial_population_size // NUM_SUBPOPULATIONS)
                ],
                Niche(specialization),
            )
            for specialization in NICHE_SPECIALIZATIONS[:NUM_SUBPOPULATIONS]
        ]
        self.environment = Environment()
        self.best_model: NanoModel = None
        self.best_fitness = float("-inf")
        self.generation = 0

    def evolve(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Evolves the swarm for one generation.

        Args:
            data (torch.Tensor): The input data.
            targets (torch.Tensor): The target values.
        """
        try:
            if not isinstance(data, torch.Tensor):
                raise TypeError("data must be a torch.Tensor")
            if not isinstance(targets, torch.Tensor):
                raise TypeError("targets must be a torch.Tensor")
            if data.size(0) != targets.size(0):
                raise ValueError("data and targets must have the same number of samples")

            self.generation += 1
            self.environment.change()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(subpop.adapt, self.environment, data, targets)
                    for subpop in self.subpopulations
                ]
                concurrent.futures.wait(futures)

            if random.random() < BOTTLENECK_PROBABILITY:
                self.bottleneck_effect()

            self.exchange_between_populations()

            all_models = [model for subpop in self.subpopulations for model in subpop.models]
            if not all_models:
                raise ValueError("No models in the population")

            best_model = max(all_models, key=lambda m: m.fitness)

            if best_model.fitness > self.best_fitness or self.best_model is None:
                self.best_fitness = best_model.fitness
                self.best_model = best_model.clone()
                print(
                    f"Generation {self.generation}: New best model found. Fitness: {self.best_fitness}"
                )

            if random.random() < EPIGENETIC_RESET_PROBABILITY:
                for subpop in self.subpopulations:
                    for model in random.sample(subpop.models, k=min(10, len(subpop.models))):
                        model.reset_epigenetic_modifications()

            self.adjust_population_size()
        except Exception as e:
            print(f"Error in evolve method: {str(e)}")
            raise
        finally:
            self.clean_memory()

    def clean_memory(self) -> None:
        """Cleans up memory by explicitly deleting models and calling garbage collection."""
        for subpop in self.subpopulations:
            for model in subpop.models:
                del model
        gc.collect()
        torch.cuda.empty_cache()

    def adjust_population_size(self) -> None:
        """Adjusts the population size to stay within the maximum limit."""
        total_population = sum(len(subpop.models) for subpop in self.subpopulations)
        if total_population > MAX_POPULATION_SIZE:
            reduction_factor = MAX_POPULATION_SIZE / total_population
            for subpop in self.subpopulations:
                subpop.models = random.sample(
                    subpop.models, int(len(subpop.models) * reduction_factor)
                )

    def bottleneck_effect(self) -> None:
        """Simulates a population bottleneck by drastically reducing the population size."""
        for subpop in self.subpopulations:
            subpop.models = subpop.models[: len(subpop.models) // 5]
            while len(subpop.models) < len(subpop.models) * 2:
                subpop.models.append(random.choice(subpop.models).clone())

    def exchange_between_populations(self) -> None:
        """Exchanges models between subpopulations to maintain diversity."""
        for i in range(len(self.subpopulations)):
            for j in range(i + 1, len(self.subpopulations)):
                if random.random() < 0.2:  # 20% chance of exchange
                    model_i = random.choice(self.subpopulations[i].models)
                    model_j = random.choice(self.subpopulations[j].models)
                    self.subpopulations[i].models.remove(model_i)
                    self.subpopulations[j].models.remove(model_j)
                    self.subpopulations[i].models.append(model_j)
                    self.subpopulations[j].models.append(model_i)

    def get_best_model(self) -> NanoModel:
        """
        Returns the best model found so far.

        Returns:
            NanoModel or SymbioticPair: The best model.
        """
        return self.best_model

    def get_population_stats(self) -> Dict[str, Any]:
        """
        Calculates and returns statistics about the current population.

        Returns:
            Dict[str, Any]: A dictionary containing population statistics.
        """
        all_models = [model for subpop in self.subpopulations for model in subpop.models]
        return {
            "total_population": len(all_models),
            "avg_fitness": np.mean([model.fitness for model in all_models]),
            "best_fitness": self.best_fitness,
            "avg_complexity": np.mean([model.get_complexity() for model in all_models]),
            "generation": self.generation,
            "mutation_rates": [subpop.mutation_rate for subpop in self.subpopulations],
            "crossover_rates": [subpop.crossover_rate for subpop in self.subpopulations],
        }

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the current state of the swarm to a checkpoint file.

        Args:
            path (str): The path to save the checkpoint file.
        """
        checkpoint = {
            "generation": self.generation,
            "best_model_state_dict": self.best_model.state_dict() if self.best_model else None,
            "best_fitness": self.best_fitness,
            "environment": self.environment.current_condition,
            "subpopulations": [
                {
                    "niche": subpop.niche.specialization,
                    "models": [model.to_json() for model in subpop.models],
                    "mutation_rate": subpop.mutation_rate,
                    "crossover_rate": subpop.crossover_rate,
                }
                for subpop in self.subpopulations
            ],
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Loads the state of the swarm from a checkpoint file.

        Args:
            path (str): The path to the checkpoint file.
        """
        checkpoint = torch.load(path)
        self.generation = checkpoint["generation"]
        if checkpoint["best_model_state_dict"]:
            self.best_model = NanoModel(self.dataset_config)
            self.best_model.load_state_dict(checkpoint["best_model_state_dict"])
        self.best_fitness = checkpoint["best_fitness"]
        self.environment.current_condition = checkpoint["environment"]

        for i, subpop_data in enumerate(checkpoint["subpopulations"]):
            self.subpopulations[i].niche.specialization = subpop_data["niche"]
            self.subpopulations[i].models = [
                (
                    NanoModel.from_json(model_json)
                    if isinstance(model_json, str)
                    else SymbioticPair.from_json(model_json)
                )
                for model_json in subpop_data["models"]
            ]
            self.subpopulations[i].mutation_rate = subpop_data["mutation_rate"]
            self.subpopulations[i].crossover_rate = subpop_data["crossover_rate"]

    def eval(self) -> None:
        """Sets all models in the swarm to evaluation mode."""
        for subpop in self.subpopulations:
            for model in subpop.models:
                model.eval()
        if self.best_model:
            self.best_model.eval()

    def train(self) -> None:
        """Sets all models in the swarm to training mode."""
        for subpop in self.subpopulations:
            for model in subpop.models:
                model.train()
        if self.best_model:
            self.best_model.train()

    def reset(self) -> None:
        """Resets the swarm to its initial state."""
        self.__init__(self.dataset_config, INITIAL_POPULATION_SIZE)

    def parallel_evolve(self, data: torch.Tensor, targets: torch.Tensor, num_processes: int = NUM_PROCESSES) -> 'EvolutionarySwarm':
        """
        Evolves the swarm in parallel.

        Args:
            data (torch.Tensor): The input data.
            targets (torch.Tensor): The target values.
            num_processes (int): The number of parallel processes to use.

        Returns:
            EvolutionarySwarm: The best swarm after parallel evolution.
        """
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [
                    executor.submit(self.evolve, data, targets) for _ in range(num_processes)
                ]
                concurrent.futures.wait(futures)

            best_swarm = max(futures, key=lambda x: x.result().best_fitness).result()
            return best_swarm
        except Exception as e:
            print(f"Error in parallel evolution: {str(e)}")
            raise
