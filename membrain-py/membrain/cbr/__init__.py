"""Case-based reasoning module for Membrain.

Provides retrieval of past experience cases, prompt building for in-context
learning, training data collection, neural classifier training, and the
parametric retriever that uses a trained model.
"""

from .prompt_builder import CasePromptBuilder
from .retriever import CaseRetriever, NonParametricRetriever
from .training_data import TrainingDataCollector, TrainingPair

__all__ = [
    "CasePromptBuilder",
    "CaseRetriever",
    "NonParametricRetriever",
    "TrainingDataCollector",
    "TrainingPair",
]

# Heavy dependencies (torch, transformers) are lazily importable:
#   from membrain.cbr.classifier import RelevanceClassifier, build_classifier
#   from membrain.cbr.dataset import CasePairDataset, CasePairCollator
#   from membrain.cbr.trainer import RetrieverTrainer, TrainingResult
#   from membrain.cbr.parametric_retriever import ParametricRetriever
