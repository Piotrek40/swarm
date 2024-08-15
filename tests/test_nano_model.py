import unittest
import torch
from models.nano_model import NanoModel, SymbioticPair
from config import DEVICE

class TestNanoModel(unittest.TestCase):
    def setUp(self):
        self.mlp_config = {
            'input_size': 10,
            'hidden_sizes': [20, 20],
            'output_size': 2,
            'problem_type': 'classification',
            'model_type': 'mlp'
        }
        self.cnn_config = {
            'input_size': (3, 32, 32),
            'hidden_sizes': [32, 64],
            'output_size': 10,
            'problem_type': 'classification',
            'model_type': 'cnn'
        }
        self.rnn_config = {
            'input_size': 50,
            'hidden_sizes': [100],
            'output_size': 5,
            'problem_type': 'sequence_labeling',
            'model_type': 'rnn'
        }

    def test_mlp_creation(self):
        model = NanoModel(self.mlp_config)
        self.assertIsInstance(model, NanoModel)
        self.assertEqual(model.layers[0].in_features, 10)
        self.assertEqual(model.layers[-1].out_features, 2)

    def test_cnn_creation(self):
        model = NanoModel(self.cnn_config)
        self.assertIsInstance(model, NanoModel)
        self.assertEqual(model.layers[0].in_channels, 3)
        self.assertEqual(model.layers[-1].out_features, 10)

    def test_rnn_creation(self):
        model = NanoModel(self.rnn_config)
        self.assertIsInstance(model, NanoModel)
        self.assertEqual(model.layers[0].input_size, 50)
        self.assertEqual(model.layers[-1].out_features, 5)

    def test_forward_pass(self):
        for config in [self.mlp_config, self.cnn_config, self.rnn_config]:
            model = NanoModel(config)
            if config['model_type'] == 'mlp':
                x = torch.randn(1, config['input_size']).to(DEVICE)
            elif config['model_type'] == 'cnn':
                x = torch.randn(1, *config['input_size']).to(DEVICE)
            else:  # rnn
                x = torch.randn(1, 10, config['input_size']).to(DEVICE)
            output = model(x)
            self.assertEqual(output.shape[1], config['output_size'])

    def test_mutate(self):
        model = NanoModel(self.mlp_config)
        initial_state = model.state_dict()
        model.mutate(0.5)
        final_state = model.state_dict()
        for key in initial_state.keys():
            self.assertFalse(torch.all(torch.eq(initial_state[key], final_state[key])))

    def test_clone(self):
        model = NanoModel(self.mlp_config)
        clone = model.clone()
        self.assertIsNot(model, clone)
        for p1, p2 in zip(model.parameters(), clone.parameters()):
            self.assertTrue(torch.all(torch.eq(p1, p2)))

    def test_symbiotic_pair(self):
        model1 = NanoModel(self.mlp_config)
        model2 = NanoModel(self.mlp_config)
        pair = SymbioticPair(model1, model2)
        x = torch.randn(1, self.mlp_config['input_size']).to(DEVICE)
        output = pair.forward(x)
        self.assertEqual(output.shape[1], self.mlp_config['output_size'])

    def test_serialization(self):
        model = NanoModel(self.mlp_config)
        json_str = model.to_json()
        loaded_model = NanoModel.from_json(json_str)
        self.assertEqual(model.get_config(), loaded_model.get_config())
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.all(torch.eq(p1, p2)))

    def test_get_complexity(self):
        model = NanoModel(self.mlp_config)
        complexity = model.get_complexity()
        self.assertIsInstance(complexity, int)
        self.assertGreater(complexity, 0)

    def test_apply_epigenetic_modification(self):
        model = NanoModel(self.mlp_config)
        model.apply_epigenetic_modification("test_modification")
        self.assertIn("test_modification", model.epigenetic_marks)

    def test_reset_epigenetic_modifications(self):
        model = NanoModel(self.mlp_config)
        model.apply_epigenetic_modification("test_modification")
        model.reset_epigenetic_modifications()
        self.assertEqual(len(model.epigenetic_marks), 0)

    def test_random_modification(self):
        model = NanoModel(self.mlp_config)
        initial_state = model.state_dict()
        model.random_modification()
        final_state = model.state_dict()
        self.assertFalse(all(torch.all(torch.eq(initial_state[key], final_state[key])) for key in initial_state.keys()))

    def test_get_l2_regularization(self):
        model = NanoModel(self.mlp_config)
        l2_reg = model.get_l2_regularization()
        self.assertIsInstance(l2_reg, torch.Tensor)
        self.assertGreater(l2_reg.item(), 0)

if __name__ == '__main__':
    unittest.main()