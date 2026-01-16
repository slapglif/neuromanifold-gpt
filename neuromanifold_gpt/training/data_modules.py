"""Data loading modules for training.

This module provides dataset and datamodule classes for training:
- MemmapDataset: Memory-mapped dataset for efficient large file handling
- TextDataModule: Lightning DataModule for preprocessed text datasets
- StreamingDataModule: Lightning DataModule for HuggingFace streaming datasets
"""

import os
import pickle
from typing import Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from loguru import logger


class MemmapDataset(Dataset):
    """Memory-mapped dataset for efficient large file handling."""

    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return max(1, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Random sampling within data (ignore idx for randomness like original)
        pos = torch.randint(len(self.data) - self.block_size, (1,)).item()
        chunk = torch.from_numpy(
            self.data[pos : pos + self.block_size + 1].astype(np.int64)
        )
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class TextDataModule(pl.LightningDataModule):
    """Lightning DataModule for text datasets."""

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        batch_size: int,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = None
        self.itos = None
        self.stoi = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Load vocab info if available
        meta_path = os.path.join(self.data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.vocab_size = meta["vocab_size"]
            self.itos = meta.get("itos")
            self.stoi = meta.get("stoi")
            logger.info(f"Loaded vocab_size={self.vocab_size} from {meta_path}")
        else:
            self.vocab_size = 50304  # GPT-2 default
            logger.info(f"No meta.pkl found, using default vocab_size={self.vocab_size}")

        self.train_ds = MemmapDataset(
            os.path.join(self.data_dir, "train.bin"), self.block_size
        )
        self.val_ds = MemmapDataset(
            os.path.join(self.data_dir, "val.bin"), self.block_size
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class StreamingDataModule(pl.LightningDataModule):
    """Lightning DataModule for streaming HuggingFace datasets."""

    def __init__(
        self,
        dataset_name: str,
        block_size: int,
        batch_size: int,
        num_workers: int = 2,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = 50257  # GPT-2 BPE tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        from neuromanifold_gpt.data.streaming import create_streaming_dataset
        self.train_ds = create_streaming_dataset(
            self.dataset_name,
            block_size=self.block_size
        )
        self.val_ds = create_streaming_dataset(
            self.dataset_name,
            block_size=self.block_size
        )
        logger.info(f"Streaming dataset: {self.dataset_name}, vocab_size={self.vocab_size}")

    def train_dataloader(self) -> DataLoader:
        from neuromanifold_gpt.data.streaming import create_streaming_dataset
        dataset = create_streaming_dataset(self.dataset_name, self.block_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        from neuromanifold_gpt.data.streaming import create_streaming_dataset
        dataset = create_streaming_dataset(self.dataset_name, self.block_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
        )
