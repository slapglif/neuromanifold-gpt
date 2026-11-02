#!/usr/bin/env python3
"""
Benchmark Attention Variants: FHN IMEX, Parallel Scan, MLA, and Standard Softmax.

Measures:
- Forward pass time
- Backward pass time
- Memory usage
- Parameter count
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from neuromanifold_gpt.model.attention.fhn import FHNAttention
from neuromanifold_gpt.model.attention.mla import MultiHeadLatentAttention
from neuromanifold_gpt.model.attention.parallel_scan import parallel_fhn_scan


class StandardSoftmaxAttention(nn.Module):
    """
    Standard multi-head scaled dot-product attention (baseline).

    This is the vanilla Transformer attention for comparison.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # QKV projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply standard multi-head attention with causal masking.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            output: Attention output of shape (B, T, D)
        """
        B, T, D = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads
        )

        # Scaled dot-product attention
        # (B, H, T, d_head) @ (B, H, d_head, T) -> (B, H, T, T)
        attn_weights = einsum(q, k, "b h t d, b h s d -> b h t s")
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Softmax and dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = einsum(attn_probs, v, "b h t s, b h s d -> b h t d")
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)

        return out


class ParallelScanAttention(nn.Module):
    """
    FHN Attention using parallel scan implementation.

    Uses FFT-based convolution for parallel dynamics computation.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # QKV projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # FHN parameters
        self.a = nn.Parameter(torch.tensor(0.7))
        self.b = nn.Parameter(torch.tensor(0.8))
        self.tau = 12.5
        self.dt = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention with parallel scan dynamics.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            output: Attention output of shape (B, T, D)
        """
        B, T, D = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads
        )

        # Compute attention scores
        attn_weights = einsum(q, k, "b h t d, b h s d -> b h t s")
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = einsum(attn_probs, v, "b h t s, b h s d -> b h t d")

        # Apply parallel scan dynamics to output
        out_flat = rearrange(out, "b h t d -> (b h) t d")
        u_init = torch.zeros_like(out_flat[:, 0, :])
        w_init = torch.zeros_like(out_flat[:, 0, :])

        u_out, _ = parallel_fhn_scan(
            u_init, w_init, out_flat, self.a, self.b, self.tau, self.dt
        )

        out = rearrange(u_out, "(b h) t d -> b h t d", b=B, h=self.n_heads)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)

        return out


def benchmark_layer(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    spectral_basis: torch.Tensor = None,
    n_iters: int = 100,
    warmup: int = 10,
):
    """Benchmark a single attention layer."""
    device = next(layer.parameters()).device
    input_tensor = input_tensor.to(device)

    # Determine if layer needs spectral_basis
    needs_spectral = isinstance(layer, FHNAttention)

    # Warmup
    for _ in range(warmup):
        if needs_spectral:
            _ = layer(input_tensor, spectral_basis)
        else:
            _ = layer(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Forward pass timing
    start = time.perf_counter()
    for _ in range(n_iters):
        if needs_spectral:
            output = layer(input_tensor, spectral_basis)
        else:
            output = layer(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / n_iters * 1000

    # Backward pass timing
    input_tensor.requires_grad_(True)
    layer.zero_grad()

    for _ in range(warmup):
        if needs_spectral:
            output = layer(input_tensor, spectral_basis)
        else:
            output = layer(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        loss = output.sum()
        loss.backward()
        layer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        if needs_spectral:
            output = layer(input_tensor, spectral_basis)
        else:
            output = layer(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        loss = output.sum()
        loss.backward()
        layer.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()
    backward_time = (time.perf_counter() - start) / n_iters * 1000

    return forward_time, backward_time


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 80)

    configs = [
        {"embed_dim": 128, "n_heads": 8, "batch_size": 4, "seq_len": 64},
        {"embed_dim": 256, "n_heads": 8, "batch_size": 2, "seq_len": 128},
        {"embed_dim": 512, "n_heads": 8, "batch_size": 1, "seq_len": 256},
    ]

    for config in configs:
        embed_dim = config["embed_dim"]
        n_heads = config["n_heads"]
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]

        print(
            f"\nConfig: embed_dim={embed_dim}, n_heads={n_heads}, "
            f"batch={batch_size}, seq_len={seq_len}"
        )
        print("-" * 80)

        x = torch.randn(batch_size, seq_len, embed_dim)

        # Create spectral basis for FHN (dummy data for benchmark)
        n_eigenvectors = 32
        spectral_basis = torch.randn(batch_size, seq_len, n_eigenvectors)

        # Initialize attention variants
        standard_attn = StandardSoftmaxAttention(embed_dim, n_heads).to(device)
        fhn_attn = FHNAttention(embed_dim, n_heads, use_imex=True, n_fhn_steps=2).to(
            device
        )
        parallel_scan_attn = ParallelScanAttention(embed_dim, n_heads).to(device)
        mla_attn = MultiHeadLatentAttention(
            embed_dim, n_heads, latent_dim=embed_dim // 8, rope_dim=8
        ).to(device)

        # Count parameters
        standard_params = count_parameters(standard_attn)
        fhn_params = count_parameters(fhn_attn)
        parallel_params = count_parameters(parallel_scan_attn)
        mla_params = count_parameters(mla_attn)

        print("\nParameters:")
        print(f"  Standard Softmax:  {standard_params:,}")
        print(
            f"  FHN IMEX:          {fhn_params:,} ({fhn_params / standard_params:.2f}x)"
        )
        print(
            f"  Parallel Scan:     {parallel_params:,} ({parallel_params / standard_params:.2f}x)"
        )
        print(
            f"  MLA:               {mla_params:,} ({mla_params / standard_params:.2f}x)"
        )

        # Benchmark each variant
        standard_fwd, standard_bwd = benchmark_layer(standard_attn, x, n_iters=100)
        fhn_fwd, fhn_bwd = benchmark_layer(fhn_attn, x, spectral_basis, n_iters=100)
        parallel_fwd, parallel_bwd = benchmark_layer(parallel_scan_attn, x, n_iters=100)
        mla_fwd, mla_bwd = benchmark_layer(mla_attn, x, n_iters=100)

        print("\nForward Pass (ms):")
        print(f"  Standard Softmax:  {standard_fwd:.3f}")
        print(f"  FHN IMEX:          {fhn_fwd:.3f} ({fhn_fwd / standard_fwd:.2f}x)")
        print(
            f"  Parallel Scan:     {parallel_fwd:.3f} ({parallel_fwd / standard_fwd:.2f}x)"
        )
        print(f"  MLA:               {mla_fwd:.3f} ({mla_fwd / standard_fwd:.2f}x)")

        print("\nBackward Pass (ms):")
        print(f"  Standard Softmax:  {standard_bwd:.3f}")
        print(f"  FHN IMEX:          {fhn_bwd:.3f} ({fhn_bwd / standard_bwd:.2f}x)")
        print(
            f"  Parallel Scan:     {parallel_bwd:.3f} ({parallel_bwd / standard_bwd:.2f}x)"
        )
        print(f"  MLA:               {mla_bwd:.3f} ({mla_bwd / standard_bwd:.2f}x)")

        print("\nTotal Time (ms):")
        standard_total = standard_fwd + standard_bwd
        fhn_total = fhn_fwd + fhn_bwd
        parallel_total = parallel_fwd + parallel_bwd
        mla_total = mla_fwd + mla_bwd

        print(f"  Standard Softmax:  {standard_total:.3f}")
        print(
            f"  FHN IMEX:          {fhn_total:.3f} ({fhn_total / standard_total:.2f}x)"
        )
        print(
            f"  Parallel Scan:     {parallel_total:.3f} ({parallel_total / standard_total:.2f}x)"
        )
        print(
            f"  MLA:               {mla_total:.3f} ({mla_total / standard_total:.2f}x)"
        )

    print("\n" + "=" * 80)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
