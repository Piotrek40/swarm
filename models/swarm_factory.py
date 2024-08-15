"""Factory for creating and managing evolutionary swarms."""

from typing import Dict, Any, List, Optional, Tuple

from torch import save, load, Tensor

from models.swarm import EvolutionarySwarm
from config import DATASET_CONFIGS, INITIAL_POPULATION_SIZE


class SwarmFactory:
    """Factory class for creating and managing evolutionary swarms."""

    def __init__(self, dataset_name: str) -> None:
        """
        Initialize the SwarmFactory.

        Args:
            dataset_name: The name of the dataset.

        Raises:
            ValueError: If the dataset is not supported.
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_config = DATASET_CONFIGS[dataset_name]
        self.metadata: Dict[str, List[Tuple[int, float]]] = {}
        self.swarm_cache: Dict[str, EvolutionarySwarm] = {}

    def create_swarm(self, population_size: int) -> EvolutionarySwarm:
        """
        Create a new evolutionary swarm or return a cached one.

        Args:
            population_size: The size of the swarm population.

        Returns:
            A new or cached evolutionary swarm.
        """
        config_key = f"{self.dataset_config['model_type']}_{population_size}"
        if config_key in self.swarm_cache:
            return self.swarm_cache[config_key].clone()

        swarm = EvolutionarySwarm(
            dataset_config=self.dataset_config, initial_population_size=population_size
        )
        self.swarm_cache[config_key] = swarm
        return swarm

    def create_swarm_for_dataset(
        self, dataset_name: str, population_size: int = INITIAL_POPULATION_SIZE
    ) -> EvolutionarySwarm:
        """
        Create a swarm for a specific dataset.

        Args:
            dataset_name: The name of the dataset.
            population_size: The size of the swarm population.

        Returns:
            A new evolutionary swarm for the specified dataset.

        Raises:
            ValueError: If the dataset is not supported.
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_config = DATASET_CONFIGS[dataset_name]
        return self.create_swarm(population_size)

    def monitor_performance(self, swarm: EvolutionarySwarm, epoch: int, val_loss: float) -> None:
        """
        Monitor the performance of a swarm.

        Args:
            swarm: The swarm to monitor.
            epoch: The current epoch.
            val_loss: The validation loss.
        """
        config = (
            f"i{self.dataset_config['input_size']}_"
            f"h{self.dataset_config['hidden_sizes']}_"
            f"o{self.dataset_config['output_size']}_"
            f"p{len(swarm.subpopulations[0].models)}"
        )
        if config not in self.metadata:
            self.metadata[config] = []
        self.metadata[config].append((epoch, val_loss))

    def get_best_config(self) -> Optional[str]:
        """
        Get the best configuration based on monitored performance.

        Returns:
            The key of the best configuration, or None if no metadata is available.
        """
        if not self.metadata:
            return None
        return min(self.metadata, key=lambda x: min(y[1] for y in self.metadata[x]))

    def save_metadata(self, path: str) -> None:
        """
        Save the metadata to a file.

        Args:
            path: The path to save the metadata.
        """
        save(self.metadata, path)

    def load_metadata(self, path: str) -> None:
        """
        Load the metadata from a file.

        Args:
            path: The path to load the metadata from.
        """
        self.metadata = load(path)

    def create_custom_swarm(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        model_type: str,
        problem_type: str,
        population_size: int,
    ) -> EvolutionarySwarm:
        """
        Create a custom swarm with specified parameters.

        Args:
            input_size: The input size of the model.
            hidden_sizes: The sizes of hidden layers.
            output_size: The output size of the model.
            model_type: The type of the model.
            problem_type: The type of the problem.
            population_size: The size of the swarm population.

        Returns:
            A new evolutionary swarm with custom configuration.
        """
        custom_config = {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "output_size": output_size,
            "model_type": model_type,
            "problem_type": problem_type,
        }
        return EvolutionarySwarm(
            dataset_config=custom_config, initial_population_size=population_size
        )


def create_swarm_for_dataset(
    dataset_name: str, population_size: int = INITIAL_POPULATION_SIZE
) -> EvolutionarySwarm:
    """
    Create a swarm for a specific dataset.

    Args:
        dataset_name: The name of the dataset.
        population_size: The size of the swarm population.

    Returns:
        A new evolutionary swarm for the specified dataset.
    """
    factory = SwarmFactory(dataset_name)
    return factory.create_swarm(population_size)


def get_swarm_factory(dataset_name: str) -> SwarmFactory:
    """
    Get a SwarmFactory instance for a specific dataset.

    Args:
        dataset_name: The name of the dataset.

    Returns:
        A SwarmFactory instance for the specified dataset.
    """
    return SwarmFactory(dataset_name)
