"""Unit tests for data loading and preprocessing utilities."""

import unittest
from typing import Tuple, List

from torch import Tensor, randn, randint
from torch.utils.data import DataLoader

from utils.data_loader import (
    load_data,
    get_dataset_info,
    get_problem_type,
    get_input_output_sizes,
    augment_image_data,
)


class TestDataLoader(unittest.TestCase):
    """Test suite for data loading and preprocessing utilities."""

    def test_load_iris_data(self) -> None:
        """Test loading of Iris dataset."""
        train_loader, val_loader, test_loader = load_data("iris")
        self._assert_loaders_not_none(train_loader, val_loader, test_loader)

    def test_load_cifar10_data(self) -> None:
        """Test loading of CIFAR10 dataset."""
        train_loader, val_loader, test_loader = load_data("cifar10")
        self._assert_loaders_not_none(train_loader, val_loader, test_loader)

    def test_load_imdb_data(self) -> None:
        """Test loading of IMDB dataset."""
        train_loader, val_loader, test_loader = load_data("imdb")
        self._assert_loaders_not_none(train_loader, val_loader, test_loader)

    def test_load_ner_data(self) -> None:
        """Test loading of NER dataset."""
        train_loader, val_loader, test_loader = load_data("ner")
        self._assert_loaders_not_none(train_loader, val_loader, test_loader)

    def test_load_ucr_timeseries_data(self) -> None:
        """Test loading of UCR TimeSeries dataset."""
        train_loader, val_loader, test_loader = load_data("ucr_timeseries")
        self._assert_loaders_not_none(train_loader, val_loader, test_loader)

    def test_get_dataset_info(self) -> None:
        """Test retrieval of dataset information."""
        info = get_dataset_info("iris")
        required_keys = ["input_size", "hidden_sizes", "output_size", "problem_type", "model_type"]
        for key in required_keys:
            self.assertIn(key, info)

    def test_get_problem_type(self) -> None:
        """Test retrieval of problem type."""
        problem_type = get_problem_type("iris")
        self.assertEqual(problem_type, "classification")

    def test_get_input_output_sizes(self) -> None:
        """Test retrieval of input and output sizes."""
        input_size, output_size = get_input_output_sizes("iris")
        self.assertEqual(input_size, 4)
        self.assertEqual(output_size, 3)

    def test_augment_image_data(self) -> None:
        """Test image data augmentation."""
        images = randn(10, 3, 32, 32)
        labels = randint(0, 10, (10,))

        augmented_images, augmented_labels = augment_image_data(images, labels)
        self.assertGreater(len(augmented_images), len(images))
        self.assertGreater(len(augmented_labels), len(labels))

    def test_invalid_dataset(self) -> None:
        """Test handling of invalid dataset name."""
        with self.assertRaises(ValueError):
            load_data("invalid_dataset")

    def test_data_shapes(self) -> None:
        """Test shapes of loaded data for all datasets."""
        datasets = ["iris", "cifar10", "imdb", "ner", "ucr_timeseries"]
        for dataset in datasets:
            with self.subTest(dataset=dataset):
                train_loader, _, _ = load_data(dataset)
                batch = next(iter(train_loader))
                self.assertEqual(len(batch), 2)  # Input and target
                self.assertEqual(batch[0].shape[0], batch[1].shape[0])  # Batch size should match

    def test_data_types(self) -> None:
        """Test types of loaded data for all datasets."""
        datasets = ["iris", "cifar10", "imdb", "ner", "ucr_timeseries"]
        for dataset in datasets:
            with self.subTest(dataset=dataset):
                train_loader, _, _ = load_data(dataset)
                batch = next(iter(train_loader))
                self.assertIsInstance(batch[0], Tensor)
                self.assertIsInstance(batch[1], Tensor)

    def _assert_loaders_not_none(self, *loaders: DataLoader) -> None:
        """Assert that all provided loaders are not None."""
        for loader in loaders:
            self.assertIsNotNone(loader)


if __name__ == "__main__":
    unittest.main()
