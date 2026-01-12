"""Streaming data loader for training without storing full datasets.

Supports:
- HuggingFace datasets with streaming=True (requires HF login)
- Local text file streaming (no auth needed)
- GPT-2 BPE tokenization for general text
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken
from typing import Optional, Iterator
from pathlib import Path
from loguru import logger
import os


class LocalTextDataset(IterableDataset):
    """Streams from local text files - no HuggingFace auth needed."""
    
    def __init__(
        self,
        path: str,
        block_size: int = 256,
        tokenizer: str = "gpt2",
    ):
        super().__init__()
        self.path = Path(path)
        self.block_size = block_size
        
        # Set up tokenizer
        self.enc = tiktoken.get_encoding(tokenizer)
        self.vocab_size = self.enc.n_vocab
        
        logger.info(f"Local text dataset: {self.path}")
        logger.info(f"Tokenizer: {tokenizer}, vocab_size: {self.vocab_size}")
        
    def __iter__(self) -> Iterator[dict]:
        """Yield tokenized chunks."""
        token_buffer = []
        
        # Handle single file or directory
        if self.path.is_file():
            files = [self.path]
        else:
            files = list(self.path.glob("**/*.txt"))
        
        for filepath in files:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    tokens = self.enc.encode_ordinary(line)
                    token_buffer.extend(tokens)
                    
                    while len(token_buffer) >= self.block_size + 1:
                        chunk = token_buffer[:self.block_size + 1]
                        token_buffer = token_buffer[self.block_size:]
                        
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield {"input_ids": x, "labels": y}


class HFStreamingDataset(IterableDataset):
    """Streams from HuggingFace datasets (requires HF token)."""
    
    DATASETS = {
        "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
        "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "text"),
        "pg19": ("deepmind/pg19", None, "text"),
    }
    
    def __init__(
        self,
        dataset_name: str = "fineweb-edu",
        block_size: int = 256,
        split: str = "train",
        tokenizer: str = "gpt2",
        buffer_size: int = 10000,
    ):
        super().__init__()
        self.block_size = block_size
        self.split = split
        self.buffer_size = buffer_size
        
        self.enc = tiktoken.get_encoding(tokenizer)
        self.vocab_size = self.enc.n_vocab
        
        if dataset_name in self.DATASETS:
            self.dataset_path, self.dataset_config, self.text_field = self.DATASETS[dataset_name]
        else:
            self.dataset_path = dataset_name
            self.dataset_config = None
            self.text_field = "text"
        
        logger.info(f"HF Streaming: {self.dataset_path}")
        
    def _get_stream(self) -> Iterator:
        from datasets import load_dataset
        ds = load_dataset(
            self.dataset_path,
            self.dataset_config,
            split=self.split,
            streaming=True,
        )
        return iter(ds.shuffle(seed=42, buffer_size=self.buffer_size))
    
    def __iter__(self) -> Iterator[dict]:
        stream = self._get_stream()
        token_buffer = []
        
        for example in stream:
            text = example.get(self.text_field, "")
            if not text:
                continue
                
            tokens = self.enc.encode_ordinary(text)
            token_buffer.extend(tokens)
            
            while len(token_buffer) >= self.block_size + 1:
                chunk = token_buffer[:self.block_size + 1]
                token_buffer = token_buffer[self.block_size:]
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}


def create_streaming_dataset(
    source: str,
    block_size: int = 256,
    tokenizer: str = "gpt2",
) -> IterableDataset:
    """Factory function to create appropriate streaming dataset.
    
    Args:
        source: Either a local path or HuggingFace dataset name
        block_size: Context window size
        tokenizer: Tokenizer name (gpt2, cl100k_base, etc.)
    
    Returns:
        IterableDataset that streams tokenized chunks
    """
    if os.path.exists(source):
        return LocalTextDataset(source, block_size, tokenizer)
    else:
        return HFStreamingDataset(source, block_size, tokenizer=tokenizer)


if __name__ == "__main__":
    print("Testing local streaming with Shakespeare...")
    
    # Use the existing shakespeare data
    shakespeare_path = "/home/mikeb/work/nano/data/shakespeare_char/input.txt"
    
    if os.path.exists(shakespeare_path):
        dataset = LocalTextDataset(shakespeare_path, block_size=128)
        print(f"Vocab size: {dataset.vocab_size}")
        
        import itertools
        for i, batch in enumerate(itertools.islice(dataset, 3)):
            print(f"Batch {i}: x.shape={batch['input_ids'].shape}")
            tokens = batch['input_ids'][:30].tolist()
            text = dataset.enc.decode(tokens)
            print(f"  Text: {text[:60]}...")
        
        print("Local streaming test passed!")
    else:
        print(f"Shakespeare data not found at {shakespeare_path}")
        print("To use HuggingFace streaming, run: huggingface-cli login")
