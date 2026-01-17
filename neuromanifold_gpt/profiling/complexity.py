import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass


@dataclass
class ComplexityMetrics:
    memory_mb: float
    flops: int
    time_ms: float
    params: int


class ComplexityProfiler:
    """
    Profile computational complexity: memory O(L) vs O(L²), FLOPs analysis.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def profile_memory(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """
        Measure peak memory usage in MB.
        """
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        with torch.no_grad():
            _ = model(input_tensor)

        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            return 0.0

    def profile_time(
        self, model: nn.Module, input_tensor: torch.Tensor, num_iterations: int = 100
    ) -> float:
        """
        Measure average forward pass time in milliseconds.
        """
        model.eval()

        if self.device == "cuda":
            torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)

        if self.device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.time() - start_time) / num_iterations
        return elapsed * 1000

    def count_flops(self, model: nn.Module, input_shape: Tuple) -> int:
        """
        Estimate FLOPs for a forward pass.
        """
        total_flops = 0

        for module in model.modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                batch_seq = input_shape[0] * input_shape[1]
                total_flops += batch_seq * in_features * out_features * 2

            elif isinstance(module, nn.Conv1d):
                kernel_size = module.kernel_size[0]
                in_channels = module.in_channels
                out_channels = module.out_channels
                output_length = input_shape[1]
                total_flops += (
                    output_length * kernel_size * in_channels * out_channels * 2
                )

        return total_flops

    def count_params(self, model: nn.Module) -> int:
        """
        Count trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def profile(
        self, model: nn.Module, input_tensor: torch.Tensor
    ) -> ComplexityMetrics:
        """
        Full complexity profile.
        """
        memory_mb = self.profile_memory(model, input_tensor)
        time_ms = self.profile_time(model, input_tensor)
        flops = self.count_flops(model, input_tensor.shape)
        params = self.count_params(model)

        return ComplexityMetrics(
            memory_mb=memory_mb, flops=flops, time_ms=time_ms, params=params
        )

    def compare_scaling(
        self, model: nn.Module, sequence_lengths: List[int], embed_dim: int = 256
    ) -> Dict[int, ComplexityMetrics]:
        """
        Compare scaling across different sequence lengths.
        """
        results = {}

        for seq_len in sequence_lengths:
            input_tensor = torch.randn(1, seq_len, embed_dim).to(self.device)
            metrics = self.profile(model, input_tensor)
            results[seq_len] = metrics

        return results


def analyze_scaling(results: Dict[int, ComplexityMetrics]) -> str:
    """
    Analyze if scaling is O(L) or O(L²).
    """
    seq_lens = sorted(results.keys())
    if len(seq_lens) < 2:
        return "Insufficient data"

    memory_ratios = []
    time_ratios = []

    for i in range(1, len(seq_lens)):
        prev_len = seq_lens[i - 1]
        curr_len = seq_lens[i]

        len_ratio = curr_len / prev_len

        mem_ratio = results[curr_len].memory_mb / results[prev_len].memory_mb
        time_ratio = results[curr_len].time_ms / results[prev_len].time_ms

        memory_ratios.append(mem_ratio / len_ratio)
        time_ratios.append(time_ratio / len_ratio)

    avg_mem_ratio = sum(memory_ratios) / len(memory_ratios)
    avg_time_ratio = sum(time_ratios) / len(time_ratios)

    if avg_mem_ratio < 1.5:
        mem_complexity = "O(L)"
    elif avg_mem_ratio < 2.5:
        mem_complexity = "O(L log L)"
    else:
        mem_complexity = "O(L²)"

    if avg_time_ratio < 1.5:
        time_complexity = "O(L)"
    elif avg_time_ratio < 2.5:
        time_complexity = "O(L log L)"
    else:
        time_complexity = "O(L²)"

    return f"Memory: {mem_complexity}, Time: {time_complexity}"
