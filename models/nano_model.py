"""Module containing NanoModel and SymbioticPair classes for flexible neural network models."""

from typing import List, Dict, Any, Tuple, Union
from torch import Tensor, device, rand, randn, norm, no_grad, tensor
from torch.nn import Module, Sequential, Linear, ReLU, Dropout, Conv2d, MaxPool2d, Flatten, LSTM
import jsonpickle

from config import DEVICE, GENERATOR, DROPOUT_RATE


class NanoModel(Module):
    """
    A flexible neural network model that can be configured as MLP, CNN, or RNN.

    Args:
        config: A dictionary containing the model configuration.

    Attributes:
        model_type: The type of the model ('mlp', 'cnn', or 'rnn').
        problem_type: The type of problem ('classification' or 'regression').
        layers: The layers of the neural network.
        epigenetic_marks: Dictionary to store epigenetic modifications.
        fitness: The fitness score of the model.
        niche: The niche of the model.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
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

        self.epigenetic_marks: Dict[str, bool] = {}
        self.fitness = float("-inf")
        self.niche = None

    def _create_mlp(self, input_size: int, hidden_sizes: List[int], output_size: int) -> Sequential:
        """Create a Multi-Layer Perceptron."""
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.extend([Linear(in_features, hidden_size), ReLU(), Dropout(DROPOUT_RATE)])
            in_features = hidden_size
        layers.append(Linear(in_features, output_size))
        return Sequential(*layers)

    def _create_cnn(self, input_size: Tuple[int, int, int], hidden_sizes: List[int], output_size: int) -> Sequential:
        """Create a Convolutional Neural Network."""
        layers = []
        in_channels = input_size[0]
        for hidden_size in hidden_sizes:
            layers.extend([
                Conv2d(in_channels, hidden_size, kernel_size=3, padding=1),
                ReLU(),
                MaxPool2d(2),
            ])
            in_channels = hidden_size
        layers.extend([
            Flatten(),
            Linear(hidden_sizes[-1] * (input_size[1] // 8) * (input_size[2] // 8), output_size),
        ])
        return Sequential(*layers)

    def _create_rnn(self, input_size: int, hidden_sizes: List[int], output_size: int) -> Sequential:
        """Create a Recurrent Neural Network."""
        return Sequential(
            LSTM(input_size, hidden_sizes[0], num_layers=len(hidden_sizes), batch_first=True),
            Linear(hidden_sizes[0], output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Define the computation performed at every call.

        Args:
            x: The input tensor.

        Returns:
            The output of the model.

        Raises:
            RuntimeError: If an error occurs during the forward pass.
        """
        try:
            if self.model_type == "rnn":
                x, _ = self.layers[0](x)
                x = self.layers[1](x[:, -1, :])
            else:
                x = self.layers(x)
            return x
        except RuntimeError as e:
            print(f"Error in forward pass: {str(e)}")
            raise

    def mutate(self, mutation_rate: float) -> None:
        """
        Apply mutation to the model parameters.

        Args:
            mutation_rate: The probability of mutating each parameter.

        Raises:
            RuntimeError: If an error occurs during mutation.
        """
        try:
            with no_grad():
                for param in self.parameters():
                    mask = (rand(param.shape, generator=GENERATOR).to(param.device) < mutation_rate)
                    param.data += randn(param.shape, generator=GENERATOR).to(param.device) * mask * 0.1
        except RuntimeError as e:
            print(f"Error during mutation: {str(e)}")
            raise

    def clone(self) -> 'NanoModel':
        """
        Create a deep copy of the model.

        Returns:
            A new instance with the same parameters and attributes.
        """
        clone = NanoModel(self.get_config())
        clone.load_state_dict(self.state_dict())
        clone.epigenetic_marks = self.epigenetic_marks.copy()
        clone.fitness = self.fitness
        clone.niche = self.niche
        return clone

    def get_complexity(self) -> int:
        """
        Calculate the complexity of the model.

        Returns:
            The total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def apply_epigenetic_modification(self, modification: str) -> None:
        """
        Apply an epigenetic modification to the model.

        Args:
            modification: The type of modification to apply.
        """
        self.epigenetic_marks[modification] = True

    def reset_epigenetic_modifications(self) -> None:
        """Reset all epigenetic modifications."""
        self.epigenetic_marks.clear()

    def random_modification(self) -> None:
        """Apply a random modification to the model parameters."""
        with no_grad():
            for param in self.parameters():
                param.data += randn_like(param) * 0.01

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the model.

        Returns:
            The model configuration.
        """
        if self.model_type == "mlp":
            input_size = self.layers[0].in_features
            hidden_sizes = [layer.out_features for layer in self.layers[:-1:3]]
            output_size = self.layers[-1].out_features
        elif self.model_type == "cnn":
            input_size = (self.layers[0].in_channels, 32, 32)  # Assuming CIFAR10
            hidden_sizes = [layer.out_channels for layer in self.layers if isinstance(layer, Conv2d)]
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
    def from_config(cls, config: Dict[str, Any]) -> 'NanoModel':
        """
        Create a new instance of NanoModel from a configuration dictionary.

        Args:
            config: The model configuration.

        Returns:
            A new instance of NanoModel.
        """
        return cls(config)

    def to_json(self) -> str:
        """
        Serialize the model to a JSON string.

        Returns:
            A JSON representation of the model.
        """
        return jsonpickle.encode(self)

    @classmethod
    def from_json(cls, json_str: str) -> 'NanoModel':
        """
        Create a new instance of NanoModel from a JSON string.

        Args:
            json_str: A JSON representation of the model.

        Returns:
            A new instance of NanoModel.
        """
        return jsonpickle.decode(json_str)

    def get_l2_regularization(self) -> Tensor:
        """
        Calculate the L2 regularization term for the model parameters.

        Returns:
            The L2 regularization term.
        """
        l2_reg = tensor(0.0, device=DEVICE)
        for param in self.parameters():
            l2_reg += norm(param)
        return l2_reg


class SymbioticPair:
    """
    Represent a pair of NanoModels that work together symbiotically.

    Args:
        model1: The first model in the pair.
        model2: The second model in the pair.

    Attributes:
        model1: The first model in the pair.
        model2: The second model in the pair.
        fitness: The fitness of the symbiotic pair.
    """

    def __init__(self, model1: NanoModel, model2: NanoModel):
        self.model1 = model1
        self.model2 = model2
        self.fitness = float("-inf")

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through both models and combine their outputs.

        Args:
            x: The input tensor.

        Returns:
            The combined output of both models.
        """
        out1 = self.model1(x)
        out2 = self.model2(x)
        return (out1 + out2) / 2  # Simple form of cooperation

    def mutate(self, mutation_rate: float) -> None:
        """
        Apply mutation to both models in the pair.

        Args:
            mutation_rate: The probability of mutating each parameter.
        """
        self.model1.mutate(mutation_rate)
        self.model2.mutate(mutation_rate)

    def get_complexity(self) -> int:
        """
        Calculate the total complexity of the symbiotic pair.

        Returns:
            The sum of complexities of both models.
        """
        return self.model1.get_complexity() + self.model2.get_complexity()

    def clone(self) -> 'SymbioticPair':
        """
        Create a deep copy of the symbiotic pair.

        Returns:
            A new instance with clones of both models.
        """
        return SymbioticPair(self.model1.clone(), self.model2.clone())

    def to(self, device: device) -> 'SymbioticPair':
        """
        Move both models to the specified device.

        Args:
            device: The device to move the models to.

        Returns:
            The symbiotic pair with models on the specified device.
        """
        self.model1.to(device)
        self.model2.to(device)
        return self

    def eval(self) -> None:
        """Set both models to evaluation mode."""
        self.model1.eval()
        self.model2.eval()

    def train(self) -> None:
        """Set both models to training mode."""
        self.model1.train()
        self.model2.train()

    def to_json(self) -> str:
        """
        Serialize the symbiotic pair to a JSON string.

        Returns:
            A JSON representation of the symbiotic pair.
        """
        return jsonpickle.encode(self)

    @classmethod
    def from_json(cls, json_str: str) -> 'SymbioticPair':
        """
        Create a new instance of SymbioticPair from a JSON string.

        Args:
            json_str: A JSON representation of the symbiotic pair.

        Returns:
            A new instance of SymbioticPair.
        """
        return jsonpickle.decode(json_str)
