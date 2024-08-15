import unittest
import torch
from utils.data_loader import load_data, get_dataset_info, get_problem_type, get_input_output_sizes, augment_image_data

class TestDataLoader(unittest.TestCase):
    def test_load_iris_data(self):
        train_loader, val_loader, test_loader = load_data('iris')
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

    def test_load_cifar10_data(self):
        train_loader, val_loader, test_loader = load_data('cifar10')
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

    def test_load_imdb_data(self):
        train_loader, val_loader, test_loader = load_data('imdb')
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

    def test_load_ner_data(self):
        train_loader, val_loader, test_loader = load_data('ner')
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

    def test_load_ucr_timeseries_data(self):
        train_loader, val_loader, test_loader = load_data('ucr_timeseries')
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

    def test_get_dataset_info(self):
        info = get_dataset_info('iris')
        self.assertIn('input_size', info)
        self.assertIn('hidden_sizes', info)
        self.assertIn('output_size', info)
        self.assertIn('problem_type', info)
        self.assertIn('model_type', info)

    def test_get_problem_type(self):
        problem_type = get_problem_type('iris')
        self.assertEqual(problem_type, 'classification')

    def test_get_input_output_sizes(self):
        input_size, output_size = get_input_output_sizes('iris')
        self.assertEqual(input_size, 4)
        self.assertEqual(output_size, 3)

    def test_augment_image_data(self):
        # Create dummy image data
        images = torch.randn(10, 3, 32, 32)
        labels = torch.randint(0, 10, (10,))

        augmented_images, augmented_labels = augment_image_data(images, labels)
        self.assertGreater(len(augmented_images), len(images))
        self.assertGreater(len(augmented_labels), len(labels))

    def test_invalid_dataset(self):
        with self.assertRaises(ValueError):
            load_data('invalid_dataset')

    def test_data_shapes(self):
        for dataset in ['iris', 'cifar10', 'imdb', 'ner', 'ucr_timeseries']:
            train_loader, _, _ = load_data(dataset)
            batch = next(iter(train_loader))
            self.assertEqual(len(batch), 2)  # Input and target
            self.assertEqual(batch[0].shape[0], batch[1].shape[0])  # Batch size should match

    def test_data_types(self):
        for dataset in ['iris', 'cifar10', 'imdb', 'ner', 'ucr_timeseries']:
            train_loader, _, _ = load_data(dataset)
            batch = next(iter(train_loader))
            self.assertIsInstance(batch[0], torch.Tensor)
            self.assertIsInstance(batch[1], torch.Tensor)

if __name__ == '__main__':
    unittest.main()