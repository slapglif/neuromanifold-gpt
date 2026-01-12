#!/usr/bin/env python3
"""
Debug FHNDynamics to find the exact bottleneck.
"""

import time
import torch
import torch.nn as nn
from rich.console import Console

console = Console()

# Tiny test
BATCH_SIZE = 2
N_HEADS = 4
N_EIG = 16
HEAD_DIM = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    console.print(f"[bold]FHN Dynamics Debug Profiler[/bold]")
    console.print(f"Device: {DEVICE}")
    console.print(f"Shape: ({BATCH_SIZE}, {N_HEADS}, {N_EIG}, {HEAD_DIM})")
    console.print()

    # Create input
    stimulus = torch.randn(BATCH_SIZE, N_HEADS, N_EIG, HEAD_DIM, device=DEVICE)

    # Import FHNDynamics
    from neuromanifold_gpt.model.attention.fhn import FHNDynamics, fhn_update_step

    fhn = FHNDynamics(dim=HEAD_DIM, tau=12.5, threshold=0.5, use_imex=True)
    fhn = fhn.to(DEVICE)
    fhn.eval()

    # Profile step by step
    console.print("Step 1: Normalize stimulus...", end=" ")
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    stim_scale = stimulus.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    stimulus_normed = stimulus / stim_scale
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    console.print(f"{(time.perf_counter() - start)*1000:.3f} ms")

    console.print("Step 2: Threshold gate...", end=" ")
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    threshold_gate = torch.sigmoid((stimulus.abs() - fhn.threshold) * 10.0)
    I = stimulus_normed * (0.1 + 0.9 * threshold_gate)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    console.print(f"{(time.perf_counter() - start)*1000:.3f} ms")

    console.print("Step 3: Initialize state...", end=" ")
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    v = torch.zeros_like(stimulus)
    w = torch.zeros_like(stimulus)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    console.print(f"{(time.perf_counter() - start)*1000:.3f} ms")

    console.print("Step 4: JIT FHN update (n_steps=2)...", end=" ")
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()

    # This is the suspected bottleneck - the JIT compiled update
    with torch.no_grad():
        v_out, w_out = fhn_update_step(
            v, w, I,
            fhn.a, fhn.b, fhn.tau, fhn.dt,
            n_steps=2
        )

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t4 = (time.perf_counter() - start) * 1000
    console.print(f"{t4:.3f} ms")

    console.print("Step 5: Scale response...", end=" ")
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    response = v_out * stim_scale
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    console.print(f"{(time.perf_counter() - start)*1000:.3f} ms")

    # Now test what happens if we call it again (JIT should be warmed up)
    console.print("\n[bold]Second call (JIT warm):[/bold]")
    console.print("JIT FHN update...", end=" ")
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        v_out2, w_out2 = fhn_update_step(
            v, w, I,
            fhn.a, fhn.b, fhn.tau, fhn.dt,
            n_steps=2
        )
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t_warm = (time.perf_counter() - start) * 1000
    console.print(f"{t_warm:.3f} ms")

    # Test full forward
    console.print("\n[bold]Full FHNDynamics forward:[/bold]")
    console.print("FHN.forward()...", end=" ")
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        response, state = fhn(stimulus, n_steps=2)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    console.print(f"{(time.perf_counter() - start)*1000:.3f} ms")

    # Estimate at full scale
    console.print("\n[bold yellow]Estimates at B=64, T=256:[/bold yellow]")
    scale = (64 / BATCH_SIZE) * (256 / N_EIG)  # N_EIG is like T in spectral space
    console.print(f"   JIT first call: ~{t4 * scale:.1f} ms (JIT compile overhead)")
    console.print(f"   JIT warm call:  ~{t_warm * scale:.1f} ms (actual runtime)")

    # Check if JIT script is the bottleneck
    console.print("\n[bold]JIT Script Analysis:[/bold]")
    console.print("The fhn_update_step is @torch.jit.script decorated.")
    console.print("First call incurs JIT compilation overhead.")
    console.print("If t_warm << t4, JIT compilation is the bottleneck.")
    if t4 > 10 * t_warm:
        console.print(f"[bold red]JIT compilation is {t4/t_warm:.1f}x slower than runtime![/bold red]")
    else:
        console.print(f"JIT overhead is {t4/t_warm:.1f}x runtime")


if __name__ == "__main__":
    main()
