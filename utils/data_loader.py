import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
from tslearn.datasets import UCR_UEA_datasets
from datasets import load_dataset
from config import DEVICE, BATCH_SIZE, TEST_SIZE, VALIDATION_SIZE, GENERATOR, DATASET_CONFIGS
import random

# Cache dla załadowanych danych
data_cache = {}

def load_data(dataset_name):
    if dataset_name in data_cache:
        return data_cache[dataset_name]

    if dataset_name == 'iris':
        loaders = load_iris_data()
    elif dataset_name == 'cifar10':
        loaders = load_cifar10_data()
    elif dataset_name == 'imdb':
        loaders = load_imdb_data()
    elif dataset_name == 'ner':
        loaders = load_synthetic_ner_data()
    elif dataset_name == 'ucr_timeseries':
        loaders = load_ucr_timeseries_data()
    elif dataset_name == 'custom':
        loaders = load_custom_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data_cache[dataset_name] = loaders
    return loaders

def load_iris_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    y = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long)).float()
    return prepare_data(X, y)

def load_cifar10_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=GENERATOR)
    val_loader, test_loader = create_val_test_loaders(test_dataset)
    
    return train_loader, val_loader, test_loader

def load_imdb_data():
    dataset = load_dataset("imdb")
    
    # Tokenizacja i tworzenie słownika
    vocab = set()
    for split in ['train', 'test']:
        for text in dataset[split]['text']:
            vocab.update(text.split())
    word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}  # 0 zarezerwowane dla paddingu
    word_to_idx['<PAD>'] = 0
    
    def tokenize(text):
        return [word_to_idx.get(word, 0) for word in text.split()[:500]]  # Ograniczenie do 500 słów
    
    # Przygotowanie danych
    train_data = [(torch.tensor(tokenize(text)), label) for text, label in zip(dataset['train']['text'], dataset['train']['label'])]
    test_data = [(torch.tensor(tokenize(text)), label) for text, label in zip(dataset['test']['text'], dataset['test']['label'])]
    
    # Podział na zbiory
    train_dataset = train_data
    val_dataset, test_dataset = train_test_split(test_data, test_size=0.5, random_state=42)
    
    # Tworzenie data loaderów
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    
    return train_loader, val_loader, test_loader

def load_synthetic_ner_data():
    # Tworzymy prosty syntetyczny zbiór danych NER
    sentences = [
        "John lives in New York",
        "Apple Inc. is based in California",
        "The Eiffel Tower is in Paris",
        "Shakespeare wrote Romeo and Juliet",
        "NASA launched a mission to Mars"
    ]
    
    tags = [
        ["B-PER", "O", "O", "B-LOC", "I-LOC"],
        ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC"],
        ["O", "B-LOC", "I-LOC", "O", "O", "B-LOC"],
        ["B-PER", "O", "B-MISC", "O", "B-MISC"],
        ["B-ORG", "O", "O", "O", "B-LOC"]
    ]
    
    words = set(word for sentence in sentences for word in sentence.split())
    word2idx = {word: idx for idx, word in enumerate(words, 1)}  # 0 zarezerwowane dla paddingu
    word2idx['<PAD>'] = 0
    tag2idx = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, "B-MISC": 7, "I-MISC": 8, "<PAD>": 9}
    
    X = [[word2idx[word] for word in sentence.split()] for sentence in sentences]
    y = [[tag2idx[tag] for tag in sentence_tags] for sentence_tags in tags]
    
    X = [torch.tensor(sentence, dtype=torch.long) for sentence in X]
    y = [torch.tensor(sentence_tags, dtype=torch.long) for sentence_tags in y]
    
    dataset = list(zip(X, y))
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    
    return train_loader, val_loader, test_loader

def load_ucr_timeseries_data():
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("ECG5000")
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    val_size = int(0.5 * len(test_dataset))
    val_dataset, test_dataset = random_split(test_dataset, [val_size, len(test_dataset) - val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_custom_data():
    # Implement custom data loading logic here
    raise NotImplementedError("Custom data loading is not implemented yet")

def prepare_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE + VALIDATION_SIZE, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE + VALIDATION_SIZE), random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=GENERATOR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader

def create_val_test_loaders(dataset):
    val_size = int(len(dataset) * VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE))
    test_size = len(dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(dataset, [val_size, test_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return val_loader, test_loader

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    max_len = max(x_lens)
    
    xx_pad = torch.full((len(xx), max_len), 0, dtype=torch.long)
    for i, x in enumerate(xx):
        xx_pad[i, :len(x)] = x
    
    yy = torch.tensor(yy)
    
    return xx_pad.to(DEVICE), yy.to(DEVICE)

def get_dataset_info(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name]

def get_problem_type(dataset_name):
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['problem_type']

def get_input_output_sizes(dataset_name):
    dataset_info = get_dataset_info(dataset_name)
    return dataset_info['input_size'], dataset_info['output_size']

def augment_image_data(images, labels):
    augmented_images = []
    augmented_labels = []
    for image, label in zip(images, labels):
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Horizontal flip
        if random.random() > 0.5:
            augmented_images.append(torch.flip(image, [2]))
            augmented_labels.append(label)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            augmented_images.append(transforms.functional.rotate(image, angle))
            augmented_labels.append(label)
    
    return torch.stack(augmented_images), torch.stack(augmented_labels)
