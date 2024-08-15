import unittest
import torch
from models.swarm import EvolutionarySwarm
from models.nano_model import NanoModel
from config import DEVICE, NUM_SUBPOPULATIONS


class TestSwarm(unittest.TestCase):
    def setUp(self):
        self.dataset_config = {
            "input_size": 10,
            "hidden_sizes": [20, 20],
            "output_size": 2,
            "problem_type": "classification",
            "model_type": "mlp",
        }
        self.swarm = EvolutionarySwarm(self.dataset_config, initial_population_size=10)

    def test_swarm_creation(self):
        self.assertEqual(len(self.swarm.subpopulations), NUM_SUBPOPULATIONS)
        for subpop in self.swarm.subpopulations:
            self.assertEqual(len(subpop.models), 10 // NUM_SUBPOPULATIONS)

    def test_evolve(self):
        data = torch.randn(32, self.dataset_config["input_size"]).to(DEVICE)
        targets = torch.randint(0, self.dataset_config["output_size"], (32,)).to(DEVICE)

        initial_best_fitness = self.swarm.best_fitness
        self.swarm.evolve(data, targets)
        self.assertGreater(self.swarm.best_fitness, initial_best_fitness)

    def test_get_best_model(self):
        best_model = self.swarm.get_best_model()
        self.assertIsInstance(best_model, NanoModel)

    def test_large_swarm(self):
        large_swarm = EvolutionarySwarm(self.dataset_config, initial_population_size=1000)
        self.assertEqual(len(large_swarm.subpopulations), NUM_SUBPOPULATIONS)
        total_models = sum(len(subpop.models) for subpop in large_swarm.subpopulations)
        self.assertEqual(total_models, 1000)

    def test_tiny_swarm(self):
        tiny_swarm = EvolutionarySwarm(self.dataset_config, initial_population_size=5)
        self.assertEqual(len(tiny_swarm.subpopulations), NUM_SUBPOPULATIONS)
        total_models = sum(len(subpop.models) for subpop in tiny_swarm.subpopulations)
        self.assertEqual(total_models, 5)

    def test_save_load_checkpoint(self):
        import tempfile
        import os

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            checkpoint_path = tmp.name

        # Save checkpoint
        self.swarm.save_checkpoint(checkpoint_path)

        # Create a new swarm and load the checkpoint
        new_swarm = EvolutionarySwarm(self.dataset_config, initial_population_size=10)
        new_swarm.load_checkpoint(checkpoint_path)

        # Check if the loaded swarm has the same properties
        self.assertEqual(self.swarm.generation, new_swarm.generation)
        self.assertEqual(self.swarm.best_fitness, new_swarm.best_fitness)

        # Clean up
        os.unlink(checkpoint_path)

    def test_parallel_evolve(self):
        data = torch.randn(32, self.dataset_config["input_size"]).to(DEVICE)
        targets = torch.randint(0, self.dataset_config["output_size"], (32,)).to(DEVICE)

        initial_best_fitness = self.swarm.best_fitness
        self.swarm.parallel_evolve(data, targets, num_processes=2)
        self.assertGreater(self.swarm.best_fitness, initial_best_fitness)

    def test_adjust_population_size(self):
        initial_population = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        for _ in range(10):  # Simulate multiple generations
            self.swarm.adjust_population_size()
        final_population = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.assertLessEqual(final_population, initial_population)

    def test_bottleneck_effect(self):
        initial_population = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.swarm.bottleneck_effect()
        bottleneck_population = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.assertLess(bottleneck_population, initial_population)

    def test_exchange_between_populations(self):
        initial_models = [model for subpop in self.swarm.subpopulations for model in subpop.models]
        self.swarm.exchange_between_populations()
        final_models = [model for subpop in self.swarm.subpopulations for model in subpop.models]
        self.assertNotEqual(initial_models, final_models)

    def test_get_population_stats(self):
        stats = self.swarm.get_population_stats()
        self.assertIn("total_population", stats)
        self.assertIn("avg_fitness", stats)
        self.assertIn("best_fitness", stats)
        self.assertIn("avg_complexity", stats)
        self.assertIn("generation", stats)
        self.assertIn("mutation_rates", stats)
        self.assertIn("crossover_rates", stats)

    def test_clean_memory(self):
        initial_models = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.swarm.clean_memory()
        final_models = sum(len(subpop.models) for subpop in self.swarm.subpopulations)
        self.assertEqual(initial_models, final_models)  # Ensure we didn't lose any models

    def test_reset(self):
        self.swarm.evolve(
            torch.randn(32, self.dataset_config["input_size"]).to(DEVICE),
            torch.randint(0, self.dataset_config["output_size"], (32,)).to(DEVICE),
        )
        initial_generation = self.swarm.generation
        self.swarm.reset()
        self.assertEqual(self.swarm.generation, 0)
        self.assertLess(self.swarm.generation, initial_generation)


if __name__ == "__main__":
    unittest.main()
