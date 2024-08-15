"""Unit tests for the EvolutionarySwarm class."""

import os
import tempfile
import unittest
from typing import Dict, Any

from torch import Tensor, randn, randint, device
from torch.nn import Module

from models.swarm import EvolutionarySwarm
from models.nano_model import NanoModel
from config import DEVICE, NUM_SUBPOPULATIONS


class TestSwarm(unittest.TestCase):
    """Test suite for the EvolutionarySwarm class."""

    def setUp(self) -> None:
        """Set up the test environment before each test method."""
        self.dataset_config: Dict[str, Any] = {
            "input_size": 10,
            "hidden_sizes": [20, 20],
            "output_size": 2,
            "problem_type": "classification",
            "model_type": "mlp",
        }
        self.swarm = EvolutionarySwarm(self.dataset_config, initial_population_size=10)

    def test_swarm_creation(self) -> None:
        """Test if the swarm is created with the correct number of subpopulations and models."""
        self.assertEqual(len(self.swarm.subpopulations), NUM_SUBPOPULATIONS)
        for subpop in self.swarm.subpopulations:
            self.assertEqual(len(subpop.models), 10 // NUM_SUBPOPULATIONS)

    def test_evolve(self) -> None:
        """Test if the swarm evolves and improves its fitness."""
        data: Tensor = randn(32, self.dataset_config["input_size"]).to(DEVICE)
        targets: Tensor = randint(0, self.dataset_config["output_size"], (32,)).to(DEVICE)

        initial_best_fitness = self.swarm.best_fitness
        self.swarm.evolve(data, targets)
        self.assertGreater(self.swarm.best_fitness, initial_best_fitness)

    def test_get_best_model(self) -> None:
        """Test if get_best_model returns a NanoModel instance."""
        best_model: Module = self.swarm.get_best_model()
        self.assertIsInstance(best_model, NanoModel)

    def test_large_swarm(self) -> None:
        """Test if a large swarm is created correctly."""
        large_swarm = EvolutionarySwarm(self.dataset_config, initial_population_size=1000)
        self.assertEqual(len(large_swarm.subpopulations), NUM_SUBPOPULATIONS)
        total_models = sum(len(subpop.models) for subpop in large_swarm.subpopulations)
        self.assertEqual(total_models, 1000)

    def test_tiny_swarm(self) -> None:
        """Test if a tiny swarm is created correctly."""
        tiny_swarm = EvolutionarySwarm(self.dataset_config, initial_population_size=5)
        self.assertEqual(len(tiny_swarm.subpopulations), NUM_SUBPOPULATIONS)
        total_models = sum(len(subpop.models) for subpop in tiny_swarm.subpopulations)
        self.assertEqual(total_models, 5)

    def test_save_load_checkpoint(self) -> None:
        """Test if the swarm can be saved to and loaded from a checkpoint."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            checkpoint_path = tmp.name

        try:
            self.swarm.save_checkpoint(checkpoint_path)

            new_swarm = EvolutionarySwarm(self.dataset_config, initial_population_size=10)
            new_swarm.load_checkpoint(checkpoint_path)

            self.assertEqual(self.swarm.generation, new_swarm.generation)
            self.assertEqual(self.swarm.best_fitness, new_swarm.best_fitness)
        finally:
            os.unlink(checkpoint_path)

    def test_parallel_evolve(self) -> None:
        """Test if parallel evolution improves the swarm's fitness."""
        data: Tensor = randn(32, self.dataset_config["input_size"]).to(DEVICE)
        targets: Tensor = randint(0, self.dataset_config["output_size"], (32,)).to(DEVICE)

        initial_best_fitness = self.swarm.best_fitness
        self.swarm.parallel_evolve(data, targets, num_processes=2)
        self.assertGreater(self.swarm.best_fitness, initial_best_fitness)

    def test_adjust_population_size(self) -> None:
        """Test if the population size is adjusted correctly."""
        initial_population = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        for _ in range(10):  # Simulate multiple generations
            self.swarm.adjust_population_size()
        final_population = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.assertLessEqual(final_population, initial_population)

    def test_bottleneck_effect(self) -> None:
        """Test if the bottleneck effect reduces the population size."""
        initial_population = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.swarm.bottleneck_effect()
        bottleneck_population = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.assertLess(bottleneck_population, initial_population)

    def test_exchange_between_populations(self) -> None:
        """Test if models are exchanged between subpopulations."""
        initial_models = [model for subpop in self.swarm.subpopulations for model in subpop.models]
        self.swarm.exchange_between_populations()
        final_models = [model for subpop in self.swarm.subpopulations for model in subpop.models]
        self.assertNotEqual(initial_models, final_models)

    def test_get_population_stats(self) -> None:
        """Test if population statistics are correctly calculated."""
        stats = self.swarm.get_population_stats()
        required_keys = [
            "total_population",
            "avg_fitness",
            "best_fitness",
            "avg_complexity",
            "generation",
            "mutation_rates",
            "crossover_rates",
        ]
        for key in required_keys:
            self.assertIn(key, stats)

    def test_clean_memory(self) -> None:
        """Test if clean_memory doesn't accidentally remove models."""
        initial_models = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.swarm.clean_memory()
        final_models = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.assertEqual(initial_models, final_models)

    def test_reset(self) -> None:
        """Test if the swarm can be reset to its initial state."""
        data: Tensor = randn(32, self.dataset_config["input_size"]).to(DEVICE)
        targets: Tensor = randint(0, self.dataset_config["output_size"], (32,)).to(DEVICE)

        self.swarm.evolve(data, targets)
        initial_generation = self.swarm.generation
        self.swarm.reset()
        self.assertEqual(self.swarm.generation, 0)
        self.assertLess(self.swarm.generation, initial_generation)


if __name__ == "__main__":
    unittest.main()
