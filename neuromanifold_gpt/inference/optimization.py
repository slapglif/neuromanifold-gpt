import torch
import torch.nn as nn
from typing import Optional


class ContinuousStateCache:
    """
    KV-cache equivalent for continuous state sequences.
    """

    def __init__(self, max_length: int = 2048, embed_dim: int = 256):
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.cache: Optional[torch.Tensor] = None
        self.position = 0

    def update(self, new_state: torch.Tensor) -> torch.Tensor:
        batch, new_len, embed_dim = new_state.shape

        if self.cache is None:
            self.cache = new_state
            self.position = new_len
            return new_state

        if self.position + new_len > self.max_length:
            shift = self.position + new_len - self.max_length
            self.cache = self.cache[:, shift:]
            self.position -= shift

        self.cache = torch.cat([self.cache, new_state], dim=1)
        self.position += new_len

        return self.cache

    def get(self) -> Optional[torch.Tensor]:
        return self.cache

    def clear(self):
        self.cache = None
        self.position = 0


class StreamingProcessor:
    """
    Streaming inference for long sequences.
    """

    def __init__(self, model: nn.Module, chunk_size: int = 512):
        self.model = model
        self.chunk_size = chunk_size
        self.cache = ContinuousStateCache()

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            full_context = self.cache.update(chunk)
            output = self.model(full_context)
            return output[:, -chunk.size(1) :, :]

    def reset(self):
        self.cache.clear()
