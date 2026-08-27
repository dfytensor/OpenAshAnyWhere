"""EnergyLM: a Transformer reinterpreted as a continuous-time energy system
trained with Equilibrium Propagation (no backpropagation).

See ``README.md`` for the full design and ``run.py`` for the experiment.
"""

from .data import CharTokenizer, DEFAULT_CORPUS, make_sequences, batch_iter
from .energy_model import EnergyLMConfig, EnergyRecurrentBlock
from .ep_trainer import EPConfig, EPTrainer, generate

__all__ = [
    "CharTokenizer",
    "DEFAULT_CORPUS",
    "make_sequences",
    "batch_iter",
    "EnergyLMConfig",
    "EnergyRecurrentBlock",
    "EPConfig",
    "EPTrainer",
    "generate",
]
