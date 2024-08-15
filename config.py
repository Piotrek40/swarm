import os
import torch

# Środowisko
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
IS_COLAB = 'COLAB_GPU' in os.environ

# Ścieżki
BASE_DIR = '/kaggle/working' if IS_KAGGLE else os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

for dir_path in [LOG_DIR, RESULT_DIR, DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Sprzęt
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_cached(0) / 1024**2:.2f} MB")

# Generator losowy
GENERATOR = torch.Generator()
GENERATOR.manual_seed(42)

# Parametry uczenia
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.3

# Parametry ewolucyjne
INITIAL_MUTATION_RATE = 0.03
MUTATION_RATE_RANGE = (0.01, 0.1)
INITIAL_CROSSOVER_RATE = 0.7
CROSSOVER_RATE_RANGE = (0.5, 0.9)
ADAPTATION_RATE = 0.05
TOURNAMENT_SIZE = 3
BOTTLENECK_PROBABILITY = 0.01
GENETIC_DRIFT_PROBABILITY = 0.005
SYMBIOSIS_PROBABILITY = 0.15
EPIGENETIC_RESET_PROBABILITY = 0.1

# Parametry populacji
INITIAL_POPULATION_SIZE = 200
MAX_POPULATION_SIZE = 1000
NUM_SUBPOPULATIONS = 5
NICHE_SPECIALIZATIONS = ['general', 'outlier_focus', 'noise_resistant', 'fast_adaptation', 'energy_efficient']
ENVIRONMENT_CONDITIONS = ['normal', 'challenging', 'extreme', 'variable']

# Parametry eksperymentów
REPORT_INTERVAL = 10
MAX_GENERATIONS = 1000
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 32
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CHECKPOINT_INTERVAL = 50

# Konfiguracje zbiorów danych
DATASET_CONFIGS = {
    'iris': {
        'input_size': 4,
        'hidden_sizes': [16, 32, 16],
        'output_size': 3,
        'problem_type': 'classification',
        'model_type': 'mlp'
    },
    'cifar10': {
        'input_size': (3, 32, 32),
        'hidden_sizes': [64, 128, 256],
        'output_size': 10,
        'problem_type': 'classification',
        'model_type': 'cnn'
    },
    'imdb': {
        'input_size': 5000,
        'hidden_sizes': [256, 128],
        'output_size': 2,
        'problem_type': 'classification',
        'model_type': 'rnn'
    },
    'ner': {
        'input_size': 100,
        'hidden_sizes': [64, 32],
        'output_size': 9,
        'problem_type': 'sequence_labeling',
        'model_type': 'rnn'
    },
    'ucr_timeseries': {
        'input_size': 140,
        'hidden_sizes': [64, 32],
        'output_size': 5,
        'problem_type': 'timeseries_classification',
        'model_type': 'rnn'
    }
}

EXPERIMENT_CONFIGS = [
    {'dataset': dataset, 'population_size': INITIAL_POPULATION_SIZE, 'generations': MAX_GENERATIONS}
    for dataset in DATASET_CONFIGS.keys()
]

# Parametry regularyzacji
L2_LAMBDA = 0.01

# Parametry równoległości
NUM_PROCESSES = 4