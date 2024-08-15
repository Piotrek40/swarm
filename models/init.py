"""
Initialization module for the models package.

This module imports and exposes the main model classes used in the project.
"""

from typing import Union

from .nano_model import NanoModel, SymbioticPair
from .swarm import EvolutionarySwarm

# Define a type alias for models that can be either NanoModel or SymbioticPair
ModelType = Union[NanoModel, SymbioticPair]

__all__ = ['NanoModel', 'SymbioticPair', 'EvolutionarySwarm', 'ModelType']
