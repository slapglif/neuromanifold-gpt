import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResolutionInvariantSampler(nn.Module):
    """
    Resolution-invariant sampling for variable-rate signals.

    Handles upsampling/downsampling between different sample rates
    (e.g., 16kHz → 44.1kHz audio) using learned interpolation.
    """

    def __init__(
        self,
        embed_dim: int,
        max_rate_ratio: float = 4.0,
        interpolation: str = "learned",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_rate_ratio = max_rate_ratio
        self.interpolation = interpolation

        if interpolation == "learned":
            self.upsample_conv = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, padding=1
            )
            self.downsample_conv = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
            )

    def upsample(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Upsample sequence to target length.

        Args:
            x: [batch, seq_len, embed_dim]
            target_length: Target sequence length

        Returns:
            upsampled: [batch, target_length, embed_dim]
        """
        batch, seq_len, embed_dim = x.shape

        if target_length == seq_len:
            return x

        if self.interpolation == "linear":
            x_permuted = x.permute(0, 2, 1)
            upsampled = F.interpolate(
                x_permuted, size=target_length, mode="linear", align_corners=False
            )
            return upsampled.permute(0, 2, 1)

        elif self.interpolation == "learned":
            x_permuted = x.permute(0, 2, 1)
            upsampled = F.interpolate(
                x_permuted, size=target_length, mode="linear", align_corners=False
            )
            upsampled = self.upsample_conv(upsampled)
            return upsampled.permute(0, 2, 1)

        else:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")

    def downsample(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Downsample sequence to target length.

        Args:
            x: [batch, seq_len, embed_dim]
            target_length: Target sequence length

        Returns:
            downsampled: [batch, target_length, embed_dim]
        """
        batch, seq_len, embed_dim = x.shape

        if target_length == seq_len:
            return x

        if self.interpolation == "linear":
            x_permuted = x.permute(0, 2, 1)
            downsampled = F.interpolate(
                x_permuted, size=target_length, mode="linear", align_corners=False
            )
            return downsampled.permute(0, 2, 1)

        elif self.interpolation == "learned":
            x_permuted = x.permute(0, 2, 1)

            stride = seq_len // target_length
            if stride > 1:
                downsampled = F.avg_pool1d(
                    x_permuted, kernel_size=stride, stride=stride
                )
                if downsampled.size(-1) != target_length:
                    downsampled = F.interpolate(
                        downsampled,
                        size=target_length,
                        mode="linear",
                        align_corners=False,
                    )
            else:
                downsampled = F.interpolate(
                    x_permuted, size=target_length, mode="linear", align_corners=False
                )

            return downsampled.permute(0, 2, 1)

        else:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")

    def resample(
        self, x: torch.Tensor, source_rate: float, target_rate: float
    ) -> torch.Tensor:
        """
        Resample from source_rate to target_rate.

        Args:
            x: [batch, seq_len, embed_dim]
            source_rate: Source sample rate (Hz)
            target_rate: Target sample rate (Hz)

        Returns:
            resampled: [batch, new_seq_len, embed_dim]
        """
        batch, seq_len, embed_dim = x.shape

        rate_ratio = target_rate / source_rate
        target_length = int(seq_len * rate_ratio)

        if rate_ratio > 1.0:
            return self.upsample(x, target_length)
        elif rate_ratio < 1.0:
            return self.downsample(x, target_length)
        else:
            return x

    def forward(
        self,
        x: torch.Tensor,
        source_rate: Optional[float] = None,
        target_rate: Optional[float] = None,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply resolution-invariant resampling.

        Args:
            x: [batch, seq_len, embed_dim]
            source_rate: Source sample rate (optional)
            target_rate: Target sample rate (optional)
            target_length: Explicit target length (optional)

        Returns:
            resampled: [batch, new_seq_len, embed_dim]
        """
        if target_length is not None:
            if target_length > x.size(1):
                return self.upsample(x, target_length)
            elif target_length < x.size(1):
                return self.downsample(x, target_length)
            else:
                return x

        elif source_rate is not None and target_rate is not None:
            return self.resample(x, source_rate, target_rate)

        else:
            return x


class AdaptiveRateEncoder(nn.Module):
    """
    Encoder that adapts to variable input rates.
    """

    def __init__(self, embed_dim: int, num_rates: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.rate_embedding = nn.Embedding(num_rates, embed_dim)

    def forward(self, x: torch.Tensor, rate_id: int) -> torch.Tensor:
        """
        Add rate-specific embedding to input.

        Args:
            x: [batch, seq_len, embed_dim]
            rate_id: Sample rate identifier (0-3)

        Returns:
            encoded: [batch, seq_len, embed_dim]
        """
        batch, seq_len, _ = x.shape

        rate_emb = self.rate_embedding(torch.tensor([rate_id], device=x.device))
        rate_emb = rate_emb.unsqueeze(1).expand(batch, seq_len, -1)

        return x + rate_emb
