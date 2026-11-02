"""
Memory Optimizer Utility

Provides automatic VRAM detection and batch size recommendation for
training on consumer GPUs (RTX 3090/4090, 8-24GB VRAM).
"""


import torch


def detect_gpu_memory() -> float:
    """
    Detect available GPU VRAM in gigabytes.

    Returns:
        float: Total GPU memory in GB, or 0.0 if no GPU available.
    """
    if not torch.cuda.is_available():
        return 0.0

    # Get total memory in bytes, convert to GB
    total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory_bytes / 1e9

    return round(total_memory_gb, 2)


def estimate_model_memory(
    n_params: int,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int = 4,  # float32 = 4 bytes
) -> float:
    """
    Estimate GPU memory usage for a model during training.

    Conservative estimation formula based on empirical measurements:
    - Model parameters: n_params * dtype_bytes * 2 (parameters + gradients)
    - Optimizer state: n_params * dtype_bytes * 2 (Adam: momentum + variance)
    - Activations: Scales with batch_size * seq_len * model_size
      For transformer models, this is approximately:
      batch_size * seq_len * hidden_dim * n_layers * 12 * dtype_bytes
      We approximate this as: batch_size * seq_len * n_params * 0.5
    - Overhead: 20% safety margin for PyTorch internals

    Args:
        n_params: Number of model parameters
        batch_size: Training batch size
        seq_len: Sequence length
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)

    Returns:
        float: Estimated memory usage in GB
    """
    # Model weights + gradients
    model_memory = n_params * dtype_bytes * 2

    # Optimizer state (Adam: momentum + variance)
    optimizer_memory = n_params * dtype_bytes * 2

    # Activations (empirical estimate for transformer models)
    # Activations scale with batch_size * seq_len and model capacity
    # For transformers: activation_memory ~= batch * seq * hidden_dim * n_layers * factor
    # Approximation: hidden_dim * n_layers ~= sqrt(n_params) * 6
    # Factor of ~12 accounts for attention, FFN intermediates, residuals, layer norms
    estimated_hidden_layers = (n_params**0.5) * 6
    activation_memory = (
        batch_size * seq_len * estimated_hidden_layers * 12 * dtype_bytes
    )

    # Total with 20% overhead for PyTorch internals, fragmentation, etc.
    total_bytes = (model_memory + optimizer_memory + activation_memory) * 1.2

    return round(total_bytes / 1e9, 2)


def recommend_batch_size(
    vram_gb: float,
    model_size_m: int,
    seq_len: int,
    dtype_bytes: int = 4,
    safety_factor: float = 0.8,
) -> int:
    """
    Recommend optimal batch size based on available VRAM.

    Uses binary search to find the largest batch size that fits in memory
    while maintaining a safety margin.

    Args:
        vram_gb: Available GPU memory in GB
        model_size_m: Model size in millions of parameters
        seq_len: Sequence length
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
        safety_factor: Use only this fraction of VRAM (default 0.8 = 80%)

    Returns:
        int: Recommended batch size (power of 2 for efficiency)
    """
    if vram_gb == 0.0:
        # No GPU, return minimal batch size
        return 1

    n_params = model_size_m * 1_000_000
    available_memory = vram_gb * safety_factor

    # Binary search for optimal batch size
    min_batch = 1
    max_batch = 1024  # Upper bound
    best_batch = 1

    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        estimated_memory = estimate_model_memory(
            n_params, mid_batch, seq_len, dtype_bytes
        )

        if estimated_memory <= available_memory:
            best_batch = mid_batch
            min_batch = mid_batch + 1
        else:
            max_batch = mid_batch - 1

    # Round down to nearest power of 2 for efficiency
    if best_batch > 1:
        power = 0
        while (1 << (power + 1)) <= best_batch:
            power += 1
        best_batch = 1 << power

    # Ensure minimum of 1
    return max(1, best_batch)


def get_memory_info() -> dict:
    """
    Get detailed GPU memory information.

    Returns:
        dict: Dictionary with memory stats (all values in GB):
            - total: Total GPU memory
            - allocated: Currently allocated memory
            - reserved: Reserved by PyTorch
            - free: Available memory
    """
    if not torch.cuda.is_available():
        return {
            "total": 0.0,
            "allocated": 0.0,
            "reserved": 0.0,
            "free": 0.0,
        }

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    free = total - allocated

    return {
        "total": round(total, 2),
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "free": round(free, 2),
    }
