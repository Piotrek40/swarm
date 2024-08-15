import torch
import torch.nn as nn
import random
import jsonpickle
from config import DEVICE, GENERATOR, DROPOUT_RATE


class NanoModel(nn.Module):
    """
    A flexible neural network model that can be configured as MLP, CNN, or RNN.

    Args:
        config (dict): A dictionary containing the model configuration.

    Attributes:
        model_type (str): The type of the model ('mlp', 'cnn', or 'rnn').
        problem_type (str): The type of problem ('classification' or 'regression').
        layers (nn.Sequential): The layers of the neural network.
    """

    def __init__(self, config):
        super(NanoModel, self).__init__()
        self.model_type = config["model_type"]
        self.problem_type = config["problem_type"]

        if self.model_type == "mlp":
            self.layers = self._create_mlp(
                config["input_size"], config["hidden_sizes"], config["output_size"]
            )
        elif self.model_type == "cnn":
            self.layers = self._create_cnn(
                config["input_size"], config["hidden_sizes"], config["output_size"]
            )
        elif self.model_type == "rnn":
            self.layers = self._create_rnn(
                config["input_size"], config["hidden_sizes"], config["output_size"]
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.to(DEVICE)

        self.epigenetic_marks = {}
        self.fitness = float("-inf")
        self.niche = None

    def _create_mlp(self, input_size, hidden_sizes, output_size):
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(in_features, hidden_size), nn.ReLU(), nn.Dropout(DROPOUT_RATE)]
            )
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        return nn.Sequential(*layers)

    def _create_cnn(self, input_size, hidden_sizes, output_size):
        layers = []
        in_channels = input_size[0]
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = hidden_size
        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(
                    hidden_sizes[-1] * (input_size[1] // 8) * (input_size[2] // 8), output_size
                ),
            ]
        )
        return nn.Sequential(*layers)

    def _create_rnn(self, input_size, hidden_sizes, output_size):
        return nn.Sequential(
            nn.LSTM(input_size, hidden_sizes[0], num_layers=len(hidden_sizes), batch_first=True),
            nn.Linear(hidden_sizes[0], output_size),
        )

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the model.
        """
        try:
            if self.model_type == "rnn":
                x, _ = self.layers[0](x)
                x = self.layers[1](x[:, -1, :])
            else:
                x = self.layers(x)
            return x
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

    def mutate(self, mutation_rate):
        """
        Applies mutation to the model parameters.

        Args:
            mutation_rate (float): The probability of mutating each parameter.
        """
        try:
            with torch.no_grad():
                for param in self.parameters():
                    mask = (
                        torch.rand(param.shape, generator=GENERATOR).to(param.device)
                        < mutation_rate
                    )
                    param.data += (
                        torch.randn(param.shape, generator=GENERATOR).to(param.device) * mask * 0.1
                    )
        except Exception as e:
            print(f"Error during mutation: {str(e)}")
            raise

    def clone(self):
        """
        Creates a deep copy of the model.

        Returns:
            NanoModel: A new instance with the same parameters and attributes.
        """
        clone = NanoModel(self.get_config())
        clone.load_state_dict(self.state_dict())
        clone.epigenetic_marks = self.epigenetic_marks.copy()
        clone.fitness = self.fitness
        clone.niche = self.niche
        return clone

    def get_complexity(self):
        """
        Calculates the complexity of the model.

        Returns:
            int: The total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def apply_epigenetic_modification(self, modification):
        """
        Applies an epigenetic modification to the model.

        Args:
            modification (str): The type of modification to apply.
        """
        self.epigenetic_marks[modification] = True

    def reset_epigenetic_modifications(self):
        """
        Resets all epigenetic modifications.
        """
        self.epigenetic_marks.clear()

    def random_modification(self):
        """
        Applies a random modification to the model parameters.
        """
        with torch.no_grad():
            for param in self.parameters():
                param.data += torch.randn_like(param) * 0.01

    def get_config(self):
        """
        Returns the configuration of the model.

        Returns:
            dict: The model configuration.
        """
        if self.model_type == "mlp":
            input_size = self.layers[0].in_features
            hidden_sizes = [layer.out_features for layer in self.layers[:-1:3]]
            output_size = self.layers[-1].out_features
        elif self.model_type == "cnn":
            input_size = (self.layers[0].in_channels, 32, 32)  # Assuming CIFAR10
            hidden_sizes = [
                layer.out_channels for layer in self.layers if isinstance(layer, nn.Conv2d)
            ]
            output_size = self.layers[-1].out_features
        else:  # RNN
            input_size = self.layers[0].input_size
            hidden_sizes = [self.layers[0].hidden_size] * self.layers[0].num_layers
            output_size = self.layers[1].out_features

        return {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "output_size": output_size,
            "problem_type": self.problem_type,
            "model_type": self.model_type,
        }

    @classmethod
    def from_config(cls, config):
        """
        Creates a new instance of NanoModel from a configuration dictionary.

        Args:
            config (dict): The model configuration.

        Returns:
            NanoModel: A new instance of NanoModel.
        """
        return cls(config)

    def to_json(self):
        """
        Serializes the model to a JSON string.

        Returns:
            str: A JSON representation of the model.
        """
        return jsonpickle.encode(self)

    @classmethod
    def from_json(cls, json_str):
        """
        Creates a new instance of NanoModel from a JSON string.

        Args:
            json_str (str): A JSON representation of the model.

        Returns:
            NanoModel: A new instance of NanoModel.
        """
        return jsonpickle.decode(json_str)

    def get_l2_regularization(self):
        """
        Calculates the L2 regularization term for the model parameters.

        Returns:
            torch.Tensor: The L2 regularization term.
        """
        l2_reg = torch.tensor(0.0, device=DEVICE)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        return l2_reg


class SymbioticPair:
    """
    Represents a pair of NanoModels that work together symbiotically.

    Args:
        model1 (NanoModel): The first model in the pair.
        model2 (NanoModel): The second model in the pair.

    Attributes:
        model1 (NanoModel): The first model in the pair.
        model2 (NanoModel): The second model in the pair.
        fitness (float): The fitness of the symbiotic pair.
    """

    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        self.fitness = float("-inf")

    def forward(self, x):
        """
        Performs a forward pass through both models and combines their outputs.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The combined output of both models.
        """
        out1 = self.model1(x)
        out2 = self.model2(x)
        return (out1 + out2) / 2  # Simple form of cooperation

    def mutate(self, mutation_rate):
        """
        Applies mutation to both models in the pair.

        Args:
            mutation_rate (float): The probability of mutating each parameter.
        """
        self.model1.mutate(mutation_rate)
        self.model2.mutate(mutation_rate)

    def get_complexity(self):
        """
        Calculates the total complexity of the symbiotic pair.

        Returns:
            int: The sum of complexities of both models.
        """
        return self.model1.get_complexity() + self.model2.get_complexity()

    def clone(self):
        """
        Creates a deep copy of the symbiotic pair.

        Returns:
            SymbioticPair: A new instance with clones of both models.
        """
        return SymbioticPair(self.model1.clone(), self.model2.clone())

    def to(self, device):
        """
        Moves both models to the specified device.

        Args:
            device (torch.device): The device to move the models to.

        Returns:
            SymbioticPair: The symbiotic pair with models on the specified device.
        """
        self.model1.to(device)
        self.model2.to(device)
        return self

    def eval(self):
        """
        Sets both models to evaluation mode.
        """
        self.model1.eval()
        self.model2.eval()

    def train(self):
        """
        Sets both models to training mode.
        """
        self.model1.train()
        self.model2.train()

    def to_json(self):
        """
        Serializes the symbiotic pair to a JSON string.

        Returns:
            str: A JSON representation of the symbiotic pair.
        """
        return jsonpickle.encode(self)

    @classmethod
    def from_json(cls, json_str):
        """
        Creates a new instance of SymbioticPair from a JSON string.

        Args:
            json_str (str): A JSON representation of the symbiotic pair.

        Returns:
            SymbioticPair: A new instance of SymbioticPair.
        """
        return jsonpickle.decode(json_str)
