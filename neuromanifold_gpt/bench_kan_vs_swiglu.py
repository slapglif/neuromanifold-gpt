#!/usr/bin/env python3
"""
Benchmark ChebyKAN vs SwiGLU FFN layers.

Measures:
- Forward pass time
- Backward pass time
- Memory usage
- Parameter count
"""

import time
import torch
import torch.nn as nn
from neuromanifold_gpt.model.kan.cheby import ChebyKANFFN
from neuromanifold_gpt.model.kan.wave import WaveKANFFN
from neuromanifold_gpt.model.block import SwiGLU


def benchmark_layer(
    layer: nn.Module, input_tensor: torch.Tensor, n_iters: int = 100, warmup: int = 10
):
    """Benchmark a single layer."""
    device = next(layer.parameters()).device
    input_tensor = input_tensor.to(device)

    for _ in range(warmup):
        _ = layer(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        output = layer(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / n_iters * 1000

    input_tensor.requires_grad_(True)
    layer.zero_grad()

    for _ in range(warmup):
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
        layer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        output = layer(input_tensor)
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
        {"embed_dim": 128, "hidden_dim": 512, "batch_size": 4, "seq_len": 64},
        {"embed_dim": 256, "hidden_dim": 1024, "batch_size": 2, "seq_len": 128},
        {"embed_dim": 512, "hidden_dim": 2048, "batch_size": 1, "seq_len": 256},
    ]

    for config in configs:
        embed_dim = config["embed_dim"]
        hidden_dim = config["hidden_dim"]
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]

        print(
            f"\nConfig: embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
            f"batch={batch_size}, seq_len={seq_len}"
        )
        print("-" * 80)

        x = torch.randn(batch_size, seq_len, embed_dim)

        swiglu = SwiGLU(embed_dim, int(hidden_dim * 2 / 3)).to(device)
        chebykan_deg3 = ChebyKANFFN(embed_dim, hidden_dim, degree=3).to(device)
        chebykan_deg4 = ChebyKANFFN(embed_dim, hidden_dim, degree=4).to(device)
        
        # WaveKAN variants
        wavekan_mexican = WaveKANFFN(embed_dim, hidden_dim, wavelet_type="mexican_hat").to(device)
        wavekan_morlet = WaveKANFFN(embed_dim, hidden_dim, wavelet_type="morlet").to(device)
        wavekan_dog = WaveKANFFN(embed_dim, hidden_dim, wavelet_type="dog").to(device)

        swiglu_params = count_parameters(swiglu)
        kan3_params = count_parameters(chebykan_deg3)
        kan4_params = count_parameters(chebykan_deg4)
        
        wave_mex_params = count_parameters(wavekan_mexican)
        wave_mor_params = count_parameters(wavekan_morlet)
        wave_dog_params = count_parameters(wavekan_dog)

        print(f"\nParameters:")
        print(f"  SwiGLU:       {swiglu_params:,}")
        print(f"  ChebyKAN d=3: {kan3_params:,} ({kan3_params / swiglu_params:.2f}x)")
        print(f"  ChebyKAN d=4: {kan4_params:,} ({kan4_params / swiglu_params:.2f}x)")
        print(f"  WaveKAN (Mex):{wave_mex_params:,} ({wave_mex_params / swiglu_params:.2f}x)")
        print(f"  WaveKAN (Mor):{wave_mor_params:,} ({wave_mor_params / swiglu_params:.2f}x)")
        print(f"  WaveKAN (DoG):{wave_dog_params:,} ({wave_dog_params / swiglu_params:.2f}x)")

        swiglu_fwd, swiglu_bwd = benchmark_layer(swiglu, x, n_iters=100)
        kan3_fwd, kan3_bwd = benchmark_layer(chebykan_deg3, x, n_iters=100)
        kan4_fwd, kan4_bwd = benchmark_layer(chebykan_deg4, x, n_iters=100)
        
        wave_mex_fwd, wave_mex_bwd = benchmark_layer(wavekan_mexican, x, n_iters=100)
        wave_mor_fwd, wave_mor_bwd = benchmark_layer(wavekan_morlet, x, n_iters=100)
        wave_dog_fwd, wave_dog_bwd = benchmark_layer(wavekan_dog, x, n_iters=100)

        print(f"\nForward Pass (ms):")
        print(f"  SwiGLU:       {swiglu_fwd:.3f}")
        print(f"  ChebyKAN d=3: {kan3_fwd:.3f} ({kan3_fwd / swiglu_fwd:.2f}x)")
        print(f"  ChebyKAN d=4: {kan4_fwd:.3f} ({kan4_fwd / swiglu_fwd:.2f}x)")
        print(f"  WaveKAN (Mex):{wave_mex_fwd:.3f} ({wave_mex_fwd / swiglu_fwd:.2f}x)")
        print(f"  WaveKAN (Mor):{wave_mor_fwd:.3f} ({wave_mor_fwd / swiglu_fwd:.2f}x)")
        print(f"  WaveKAN (DoG):{wave_dog_fwd:.3f} ({wave_dog_fwd / swiglu_fwd:.2f}x)")

        print(f"\nBackward Pass (ms):")
        print(f"  SwiGLU:       {swiglu_bwd:.3f}")
        print(f"  ChebyKAN d=3: {kan3_bwd:.3f} ({kan3_bwd / swiglu_bwd:.2f}x)")
        print(f"  ChebyKAN d=4: {kan4_bwd:.3f} ({kan4_bwd / swiglu_bwd:.2f}x)")
        print(f"  WaveKAN (Mex):{wave_mex_bwd:.3f} ({wave_mex_bwd / swiglu_bwd:.2f}x)")
        print(f"  WaveKAN (Mor):{wave_mor_bwd:.3f} ({wave_mor_bwd / swiglu_bwd:.2f}x)")
        print(f"  WaveKAN (DoG):{wave_dog_bwd:.3f} ({wave_dog_bwd / swiglu_bwd:.2f}x)")

        print(f"\nTotal Time (ms):")
        swiglu_total = swiglu_fwd + swiglu_bwd
        kan3_total = kan3_fwd + kan3_bwd
        kan4_total = kan4_fwd + kan4_bwd
        wave_mex_total = wave_mex_fwd + wave_mex_bwd
        wave_mor_total = wave_mor_fwd + wave_mor_bwd
        wave_dog_total = wave_dog_fwd + wave_dog_bwd
        
        print(f"  SwiGLU:       {swiglu_total:.3f}")
        print(f"  ChebyKAN d=3: {kan3_total:.3f} ({kan3_total / swiglu_total:.2f}x)")
        print(f"  ChebyKAN d=4: {kan4_total:.3f} ({kan4_total / swiglu_total:.2f}x)")
        print(f"  WaveKAN (Mex):{wave_mex_total:.3f} ({wave_mex_total / swiglu_total:.2f}x)")
        print(f"  WaveKAN (Mor):{wave_mor_total:.3f} ({wave_mor_total / swiglu_total:.2f}x)")
        print(f"  WaveKAN (DoG):{wave_dog_total:.3f} ({wave_dog_total / swiglu_total:.2f}x)")

    print("\n" + "=" * 80)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
