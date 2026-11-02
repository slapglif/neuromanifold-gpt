import torch
import torch.nn as nn


def karmarkar_karp_partition(
    values: torch.Tensor, k_partitions: int = 2
) -> torch.Tensor:
    """
    Partition values into k sets with approximately equal sums (number partitioning).
    Uses the Karmarkar-Karp differencing heuristic (largest first).

    This is used to balance spectral inputs to FHN solvers, minimizing variance
    and preventing "stiffness" (rogue waves).

    Args:
        values: (..., N) tensor of values to partition
        k_partitions: Number of partitions (currently supports 2 via differencing)

    Returns:
        indices: (..., N) partition assignment indices (0 or 1)

    Note: For GPU efficiency, we use a vectorized sorting approach ("folding").
    We sort values, then pair largest with smallest (folding).
    This approximates the equal-sum partition for 2 sets.
    """
    # 1. Sort values descending
    # values: (B, H, N)
    sorted_vals, indices = torch.sort(values, dim=-1, descending=True)

    # 2. Assign to sets
    # Simple folding heuristic:
    # Set 0: indices [0, 3, 4, 7, ...]
    # Set 1: indices [1, 2, 5, 6, ...]
    # This pattern (ABBA) minimizes the difference of prefix sums better than alternating (ABAB).

    N = values.shape[-1]
    torch.zeros_like(indices, dtype=torch.long)

    # Create ABBA pattern mask
    # 0, 1, 1, 0, 0, 1, 1, 0...
    arange = torch.arange(N, device=values.device)
    mask_b = (arange % 4 == 1) | (arange % 4 == 2)

    # Assign sorted indices to sets
    # We want to map original indices to partition IDs
    # indices maps: position_in_sorted -> original_index
    # We want: original_index -> partition_id

    # assignment[original_index] = partition_id
    # We can do this by scattering
    partition_ids = mask_b.long()  # (N,) of 0s and 1s

    # Expand to batch/head dims if needed
    if values.ndim > 1:
        partition_ids = partition_ids.expand_as(values)

    # Scatter partition_ids back to original order
    # result[b, h, indices[b, h, i]] = partition_ids[b, h, i]
    final_assignment = torch.zeros_like(indices, dtype=torch.long)
    final_assignment.scatter_(dim=-1, index=indices, src=partition_ids)

    return final_assignment


class SpectralPartitioner(nn.Module):
    def __init__(self, n_eigenvectors: int):
        super().__init__()
        self.n = n_eigenvectors

    def forward(
        self, spectral_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fold spectral input into 2 balanced streams.

        Args:
            spectral_input: (B, H, K)

        Returns:
            folded_input: (B, H, K/2) - Sum of paired inputs
            indices: (B, H, K) - Permutation to restore order
        """
        B, H, K = spectral_input.shape
        assert K % 2 == 0, "K must be even for folding"

        # Sort by magnitude to identify heavy hitters
        # We process 'energy' (abs value) for partitioning decision,
        # but sum the actual signed values.
        magnitudes = spectral_input.abs()
        _, indices = torch.sort(magnitudes, dim=-1, descending=True)

        # Gather values in sorted order
        sorted_input = torch.gather(spectral_input, dim=-1, index=indices)

        # Fold: Pair largest (0) with smallest (K-1), 2nd largest (1) with 2nd smallest (K-2)...
        # This is slightly different from ABBA but better for "cancelling out" extreme values if they have opposite signs,
        # or balancing total energy if they have same signs.
        # For FHN, we want to balance the *magnitude* of the input current I.
        # So pairing large I with small I is good.

        half = K // 2
        top_half = sorted_input[..., :half]
        bottom_half = sorted_input[..., half:]

        # Reverse bottom half to pair Largest with Smallest
        bottom_half_flipped = bottom_half.flip(dims=[-1])

        # Sum them to get inputs for the FHN solvers
        # We now have K/2 solvers instead of K
        folded_input = top_half + bottom_half_flipped

        return folded_input, indices

    def unfold(
        self, folded_output: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Distribute FHN response back to original spectral modes.
        Both "partners" in the fold receive the SAME response (entangled dynamics).
        """
        # folded_output: (B, H, K/2)

        # We need to reconstruct (B, H, K)
        # top_half_response = folded_output
        # bottom_half_response = folded_output (flipped back)

        response_top = folded_output
        response_bottom = folded_output.flip(dims=[-1])

        sorted_response = torch.cat([response_top, response_bottom], dim=-1)

        # Scatter back to original positions
        original_response = torch.zeros_like(sorted_response)
        original_response.scatter_(dim=-1, index=indices, src=sorted_response)

        return original_response
