import torch
from models.swarm import EvolutionarySwarm
from config import DEVICE, DATASET_CONFIGS, INITIAL_POPULATION_SIZE


class SwarmFactory:
    def __init__(self, dataset_name):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_config = DATASET_CONFIGS[dataset_name]
        self.metadata = {}
        self.swarm_cache = {}

    def create_swarm(self, population_size):
        config_key = f"{self.dataset_config['model_type']}_{population_size}"
        if config_key in self.swarm_cache:
            return self.swarm_cache[config_key].clone()

        swarm = EvolutionarySwarm(
            dataset_config=self.dataset_config, initial_population_size=population_size
        )
        self.swarm_cache[config_key] = swarm
        return swarm

    def create_swarm_for_dataset(self, dataset_name, population_size=INITIAL_POPULATION_SIZE):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_config = DATASET_CONFIGS[dataset_name]
        return self.create_swarm(population_size)

    def monitor_performance(self, swarm, epoch, val_loss):
        config = f"i{self.dataset_config['input_size']}_h{self.dataset_config['hidden_sizes']}_o{self.dataset_config['output_size']}_p{len(swarm.subpopulations[0].models)}"
        if config not in self.metadata:
            self.metadata[config] = []
        self.metadata[config].append((epoch, val_loss))

    def get_best_config(self):
        if not self.metadata:
            return None
        return min(self.metadata, key=lambda x: min(y[1] for y in self.metadata[x]))

    def save_metadata(self, path):
        torch.save(self.metadata, path)

    def load_metadata(self, path):
        self.metadata = torch.load(path)

    def create_custom_swarm(
        self, input_size, hidden_sizes, output_size, model_type, problem_type, population_size
    ):
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


def create_swarm_for_dataset(dataset_name, population_size=INITIAL_POPULATION_SIZE):
    factory = SwarmFactory(dataset_name)
    return factory.create_swarm(population_size)


def get_swarm_factory(dataset_name):
    return SwarmFactory(dataset_name)
