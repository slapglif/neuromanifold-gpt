"""Training components for NeuroManifoldGPT.

This module exports training infrastructure:

Configuration:
    TrainConfig: Training hyperparameters and settings

Data Modules:
    TextDataModule: Lightning DataModule for local memmap datasets
    StreamingDataModule: Lightning DataModule for HuggingFace streaming datasets
    MemmapDataset: Memory-mapped dataset for efficient large file handling

Lightning Module:
    NeuroManifoldLitModule: PyTorch Lightning wrapper with training/val loops

Callbacks:
    SampleGenerationCallback: Periodic text generation during training
    MFUCallback: Model FLOPs Utilization tracking

Training:
    train: Main training orchestration function
"""

# Configuration
from neuromanifold_gpt.training.config import TrainConfig

# Data modules
from neuromanifold_gpt.training.data_modules import (
    MemmapDataset,
    TextDataModule,
    StreamingDataModule,
)

# Lightning module
from neuromanifold_gpt.training.lightning_module import NeuroManifoldLitModule

# Callbacks
from neuromanifold_gpt.training.callbacks import (
    SampleGenerationCallback,
    MFUCallback,
)

# Training orchestration
from neuromanifold_gpt.training.trainer import train

__all__ = [
    # Configuration
    "TrainConfig",
    # Data modules
    "MemmapDataset",
    "TextDataModule",
    "StreamingDataModule",
    # Lightning module
    "NeuroManifoldLitModule",
    # Callbacks
    "SampleGenerationCallback",
    "MFUCallback",
    # Training
    "train",
]
