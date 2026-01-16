
import torch
import time
from neuromanifold_gpt.model.mhc import HyperConnections, sinkhorn_log

try:
    from neuromanifold_gpt.model.mhc_fused import fused_mhc_width_connection
    has_triton = True
except (ImportError, RuntimeError):
    has_triton = False


def benchmark_mhc_fusion():
    """Benchmark Fused Triton vs PyTorch mHC width_connection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" or not has_triton:
        print("Skipping fusion benchmark (CUDA or Triton not available)")
        return

    # Test with different stream counts
    test_configs = [
        {"B": 32, "T": 256, "S": 2, "D": 384},
        {"B": 32, "T": 256, "S": 4, "D": 384},
        {"B": 32, "T": 256, "S": 8, "D": 384},
    ]

    print("=" * 80)
    print("mHC Width Connection Fusion Benchmark")
    print("=" * 80)

    for config in test_configs:
        B, T, S, D = config["B"], config["T"], config["S"], config["D"]

        # Create HyperConnections for unfused path
        hc = HyperConnections(S, dim=D).to(device)

        # Create input: (B*S, T, D)
        residuals = torch.randn(B * S, T, D, device=device)

        # Precompute matrices for fused path
        with torch.no_grad():
            h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
            h_pre = hc.H_pre_logits.softmax(dim=-1)

        # Warmup
        for _ in range(10):
            # Unfused
            with torch.no_grad():
                hc.width_connection(residuals)
            # Fused
            with torch.no_grad():
                fused_mhc_width_connection(residuals, h_res, h_pre)

        torch.cuda.synchronize()

        # Benchmark Unfused (PyTorch)
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                hc.width_connection(residuals)
        torch.cuda.synchronize()
        time_unfused = (time.time() - start) * 1000 / 100

        # Benchmark Fused (Triton)
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                fused_mhc_width_connection(residuals, h_res, h_pre)
        torch.cuda.synchronize()
        time_fused = (time.time() - start) * 1000 / 100

        print(f"\nConfig: B={B}, T={T}, S={S}, D={D}")
        print(f"Unfused (PyTorch): {time_unfused:.3f} ms")
        print(f"Fused (Triton):    {time_fused:.3f} ms")
        print(f"Speedup:           {time_unfused / time_fused:.2f}x")

        # Verify correctness
        with torch.no_grad():
            branch_unfused, residuals_out_unfused, _ = hc.width_connection(residuals)
            branch_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

            branch_match = torch.allclose(branch_unfused, branch_fused, rtol=1e-4, atol=1e-5)
            residuals_match = torch.allclose(residuals_out_unfused, residuals_out_fused, rtol=1e-4, atol=1e-5)

            if branch_match and residuals_match:
                print("Correctness:       ✓ PASS")
            else:
                print(f"Correctness:       ✗ FAIL (branch={branch_match}, residuals={residuals_match})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_mhc_fusion()
