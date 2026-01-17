from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class GradientCheckpointing(nn.Module):
    """
    Gradient checkpointing for memory-efficient training on long sequences.

    Trades compute for memory by recomputing activations during backward pass.
    """

    def __init__(self, module: nn.Module, num_segments: int = 4):
        super().__init__()
        self.module = module
        self.num_segments = num_segments

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.training:
            return self.module(x, *args, **kwargs)

        return checkpoint(self.module, x, *args, **kwargs, use_reentrant=False)


class GradientAccumulator:
    """
    Gradient accumulation for training with effective large batch sizes.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        max_grad_norm: Optional[float] = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0

    def step(self, loss: torch.Tensor):
        """
        Accumulate gradients and step optimizer when accumulation_steps is reached.
        """
        loss = loss / self.accumulation_steps
        loss.backward()

        self.current_step += 1

        if self.current_step % self.accumulation_steps == 0:
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.current_step = 0


class SequenceChunker:
    """
    Chunk long sequences for memory-efficient processing.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, x: torch.Tensor) -> list:
        """
        Split sequence into overlapping chunks.

        Args:
            x: [batch, seq_len, ...]

        Returns:
            List of chunks [batch, chunk_size, ...]
        """
        batch, seq_len = x.shape[:2]
        chunks = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end]
            chunks.append(chunk)

            if end == seq_len:
                break

            start = end - self.overlap

        return chunks

    def merge(self, chunks: list, original_length: int) -> torch.Tensor:
        """
        Merge overlapping chunks back into sequence.
        """
        if len(chunks) == 1:
            return chunks[0]

        batch = chunks[0].shape[0]
        embed_dim = chunks[0].shape[-1]

        merged = torch.zeros(batch, original_length, embed_dim, device=chunks[0].device)
        weights = torch.zeros(batch, original_length, 1, device=chunks[0].device)

        start = 0
        for chunk in chunks:
            chunk_len = chunk.shape[1]
            end = start + chunk_len

            merged[:, start:end] += chunk
            weights[:, start:end] += 1.0

            start = end - self.overlap

        merged = merged / weights.clamp(min=1.0)
        return merged


class LongSequenceTrainer:
    """
    Complete training infrastructure for long sequences (>100k tokens).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chunk_size: int = 2048,
        accumulation_steps: int = 8,
        use_checkpointing: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.chunker = SequenceChunker(chunk_size=chunk_size, overlap=128)
        self.accumulator = GradientAccumulator(
            model, optimizer, accumulation_steps, max_grad_norm
        )
        self.use_checkpointing = use_checkpointing

        if use_checkpointing:
            self._wrap_with_checkpointing()

    def _wrap_with_checkpointing(self):
        """
        Wrap transformer blocks with gradient checkpointing.
        """
        for name, module in self.model.named_modules():
            if "block" in name.lower() or "layer" in name.lower():
                if isinstance(module, nn.Module) and len(list(module.children())) > 0:
                    parent_module = self._get_parent_module(name)
                    if parent_module is not None:
                        child_name = name.split(".")[-1]
                        setattr(
                            parent_module,
                            child_name,
                            GradientCheckpointing(module),
                        )

    def _get_parent_module(self, name: str) -> Optional[nn.Module]:
        parts = name.split(".")
        if len(parts) < 2:
            return None

        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent

    def train_step(self, x: torch.Tensor, targets: torch.Tensor, loss_fn: Callable):
        """
        Single training step with chunking and accumulation.
        """
        chunks = self.chunker.chunk(x)
        target_chunks = self.chunker.chunk(targets)

        total_loss = 0.0

        for chunk, target_chunk in zip(chunks, target_chunks):
            outputs = self.model(chunk)
            loss = loss_fn(outputs, target_chunk)

            self.accumulator.step(loss)
            total_loss += loss.item()

        return total_loss / len(chunks)
