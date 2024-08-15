"""
Configuration module for NanoAI project.

This module contains all the configuration variables and constants used throughout the project.
"""

import os
from typing import Dict, List, Tuple, Any

from torch import device, cuda, Generator
from torch.cuda import get_device_name, memory_allocated, memory_cached

# Environment
IS_KAGGLE: bool = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_COLAB: bool = "COLAB_GPU" in os.environ

# Paths
BASE_DIR: str = "/kaggle/working" if IS_KAGGLE else os.path.dirname(os.path.abspath(__file__))
LOG_DIR: str = os.path.join(BASE_DIR, "logs")
RESULT_DIR: str = os.path.join(BASE_DIR, "results")
DATA_DIR: str = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")

for dir_path in [LOG_DIR, RESULT_DIR, DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Hardware
DEVICE: device = device("cuda" if cuda.is_available() else "cpu")
USE_AMP: bool = cuda.is_available()

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {get_device_name(0)}")
    print(f"Memory Allocated: {memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory Cached: {memory_cached(0) / 1024**2:.2f} MB")

# Random generator
GENERATOR: Generator = Generator()
GENERATOR.manual_seed(42)

# Learning parameters
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 0.0001
DROPOUT_RATE: float = 0.3

# Evolutionary parameters
INITIAL_MUTATION_RATE: float = 0.03
MUTATION_RATE_RANGE: Tuple[float, float] = (0.01, 0.1)
INITIAL_CROSSOVER_RATE: float = 0.7
CROSSOVER_RATE_RANGE: Tuple[float, float] = (0.5, 0.9)
ADAPTATION_RATE: float = 0.05
TOURNAMENT_SIZE: int = 3
BOTTLENECK_PROBABILITY: float = 0.01
GENETIC_DRIFT_PROBABILITY: float = 0.005
SYMBIOSIS_PROBABILITY: float = 0.15
EPIGENETIC_RESET_PROBABILITY: float = 0.1

# Population parameters
INITIAL_POPULATION_SIZE: int = 200
MAX_POPULATION_SIZE: int = 1000
NUM_SUBPOPULATIONS: int = 5
NICHE_SPECIALIZATIONS: List[str] = [
    "general",
    "outlier_focus",
    "noise_resistant",
    "fast_adaptation",
    "energy_efficient",
]
ENVIRONMENT_CONDITIONS: List[str] = ["normal", "challenging", "extreme", "variable"]

# Experiment parameters
REPORT_INTERVAL: int = 10
MAX_GENERATIONS: int = 1000
EARLY_STOPPING_PATIENCE: int = 20
BATCH_SIZE: int = 32
TEST_SIZE: float = 0.2
VALIDATION_SIZE: float = 0.2
CHECKPOINT_INTERVAL: int = 50

# Dataset configurations
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "iris": {
        "input_size": 4,
        "hidden_sizes": [16, 32, 16],
        "output_size": 3,
        "problem_type": "classification",
        "model_type": "mlp",
    },
    "cifar10": {
        "input_size": (3, 32, 32),
        "hidden_sizes": [64, 128, 256],
        "output_size": 10,
        "problem_type": "classification",
        "model_type": "cnn",
    },
    "imdb": {
        "input_size": 5000,
        "hidden_sizes": [256, 128],
        "output_size": 2,
        "problem_type": "classification",
        "model_type": "rnn",
    },
    "ner": {
        "input_size": 100,
        "hidden_sizes": [64, 32],
        "output_size": 9,
        "problem_type": "sequence_labeling",
        "model_type": "rnn",
    },
    "ucr_timeseries": {
        "input_size": 140,
        "hidden_sizes": [64, 32],
        "output_size": 5,
        "problem_type": "timeseries_classification",
        "model_type": "rnn",
    },
}

EXPERIMENT_CONFIGS: List[Dict[str, Any]] = [
    {"dataset": dataset, "population_size": INITIAL_POPULATION_SIZE, "generations": MAX_GENERATIONS}
    for dataset in DATASET_CONFIGS.keys()
]

# Regularization parameters
L2_LAMBDA: float = 0.01

# Parallelism parameters
NUM_PROCESSES: int = 4
