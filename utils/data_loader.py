"""Data loading and preprocessing utilities for NanoAI."""

from typing import Tuple, List, Dict, Any
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, tensor, long, float32, nn, flip, stack
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from tslearn.datasets import UCR_UEA_datasets
from datasets import load_dataset

from config import DEVICE, BATCH_SIZE, TEST_SIZE, VALIDATION_SIZE, GENERATOR, DATASET_CONFIGS

# Cache for loaded data
data_cache: Dict[str, Tuple[DataLoader, DataLoader, DataLoader]] = {}


def load_data(dataset_name: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and prepare data for a given dataset.

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        Tuple of train, validation, and test data loaders.

    Raises:
        ValueError: If the dataset is not supported.
    """
    if dataset_name in data_cache:
        return data_cache[dataset_name]

    loaders = {
        "iris": load_iris_data,
        "cifar10": load_cifar10_data,
        "imdb": load_imdb_data,
        "ner": load_synthetic_ner_data,
        "ucr_timeseries": load_ucr_timeseries_data,
        "custom": load_custom_data,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data_loaders = loaders[dataset_name]()
    data_cache[dataset_name] = data_loaders
    return data_loaders


def load_iris_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare the Iris dataset."""
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    y = nn.functional.one_hot(tensor(y, dtype=long)).float()
    return prepare_data(X, y)


def load_cifar10_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare the CIFAR10 dataset."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=GENERATOR
    )
    val_loader, test_loader = create_val_test_loaders(test_dataset)

    return train_loader, val_loader, test_loader


def load_imdb_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare the IMDB dataset."""
    dataset = load_dataset("imdb")

    vocab = set(
        word
        for split in ["train", "test"]
        for text in dataset[split]["text"]
        for word in text.split()
    )
    word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}
    word_to_idx["<PAD>"] = 0

    def tokenize(text: str) -> List[int]:
        return [word_to_idx.get(word, 0) for word in text.split()[:500]]

    train_data = [
        (tensor(tokenize(text)), label)
        for text, label in zip(dataset["train"]["text"], dataset["train"]["label"])
    ]
    test_data = [
        (tensor(tokenize(text)), label)
        for text, label in zip(dataset["test"]["text"], dataset["test"]["label"])
    ]

    train_dataset = train_data
    val_dataset, test_dataset = train_test_split(test_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
    )

    return train_loader, val_loader, test_loader


def load_synthetic_ner_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare synthetic NER data."""
    sentences = [
        "John lives in New York",
        "Apple Inc. is based in California",
        "The Eiffel Tower is in Paris",
        "Shakespeare wrote Romeo and Juliet",
        "NASA launched a mission to Mars",
    ]

    tags = [
        ["B-PER", "O", "O", "B-LOC", "I-LOC"],
        ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC"],
        ["O", "B-LOC", "I-LOC", "O", "O", "B-LOC"],
        ["B-PER", "O", "B-MISC", "O", "B-MISC"],
        ["B-ORG", "O", "O", "O", "B-LOC"],
    ]

    words = set(word for sentence in sentences for word in sentence.split())
    word2idx = {word: idx for idx, word in enumerate(words, 1)}
    word2idx["<PAD>"] = 0
    tag2idx = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-LOC": 3,
        "I-LOC": 4,
        "B-ORG": 5,
        "I-ORG": 6,
        "B-MISC": 7,
        "I-MISC": 8,
        "<PAD>": 9,
    }

    X = [
        tensor([word2idx[word] for word in sentence.split()], dtype=long) for sentence in sentences
    ]
    y = [tensor([tag2idx[tag] for tag in sentence_tags], dtype=long) for sentence_tags in tags]

    dataset = list(zip(X, y))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
    )

    return train_loader, val_loader, test_loader


def load_ucr_timeseries_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare UCR time series data."""
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("ECG5000")

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = tensor(X_train, dtype=float32)
    y_train = tensor(y_train, dtype=long)
    X_test = tensor(X_test, dtype=float32)
    y_test = tensor(y_test, dtype=long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    val_size = int(0.5 * len(test_dataset))
    val_dataset, test_dataset = random_split(test_dataset, [val_size, len(test_dataset) - val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def load_custom_data() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare custom data."""
    raise NotImplementedError("Custom data loading is not implemented yet")


def prepare_data(X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data for training, validation, and testing.

    Args:
        X: Input features.
        y: Target values.

    Returns:
        Tuple of train, validation, and test data loaders.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VALIDATION_SIZE, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE), random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train, X_val, X_test = map(lambda x: tensor(x, dtype=float32), (X_train, X_val, X_test))
    y_train, y_val, y_test = map(lambda x: tensor(x, dtype=float32), (y_train, y_val, y_test))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=GENERATOR
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader


def create_val_test_loaders(dataset: TensorDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Create validation and test data loaders from a single dataset.

    Args:
        dataset: The dataset to split.

    Returns:
        Tuple of validation and test data loaders.
    """
    val_size = int(len(dataset) * VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE))
    test_size = len(dataset) - val_size
    val_dataset, test_dataset = random_split(dataset, [val_size, test_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return val_loader, test_loader


def pad_collate(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """
    Pad and collate data for variable length sequences.

    Args:
        batch: A batch of data.

    Returns:
        Tuple of padded input sequences and labels.
    """
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    max_len = max(x_lens)

    xx_pad = tensor([[0] * max_len for _ in range(len(xx))], dtype=long)
    for i, x in enumerate(xx):
        xx_pad[i, : len(x)] = x

    yy = tensor(yy)

    return xx_pad.to(DEVICE), yy.to(DEVICE)


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset configuration information.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Dictionary containing dataset configuration.

    Raises:
        ValueError: If the dataset is not supported.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name]


def get_problem_type(dataset_name: str) -> str:
    """
    Get problem type for a given dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Problem type (e.g., 'classification', 'regression').
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["problem_type"]


def get_input_output_sizes(dataset_name: str) -> Tuple[int, int]:
    """
    Get input and output sizes for a given dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Tuple of input size and output size.
    """
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info["input_size"], dataset_info["output_size"]


def augment_image_data(images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Augment image data with flips and rotations.

    Args:
        images: Input images.
        labels: Corresponding labels.

    Returns:
        Tuple of augmented images and labels.
    """
    augmented_images = []
    augmented_labels = []
    for image, label in zip(images, labels):
        augmented_images.append(image)
        augmented_labels.append(label)

        if random.random() > 0.5:
            augmented_images.append(flip(image, [2]))
            augmented_labels.append(label)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            augmented_images.append(transforms.functional.rotate(image, angle))
            augmented_labels.append(label)

    return stack(augmented_images), stack(augmented_labels)
