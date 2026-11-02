# neuromanifold_gpt/model/kan/faster/utils.py
"""
Utility functions for FasterKAN.

Includes helper for replacing nn.Linear layers with FasterKANLinear.
"""

import torch.nn as nn

from .linear import FasterKANLinear


def replace_linear_with_fasterkan(
    module: nn.Module,
    num_centers: int = 4,
    skip_names: set[str] | None = None,
    skip_module_types: tuple | None = None,
    _prefix: str = "",
) -> nn.Module:
    """Recursively replace nn.Linear with FasterKANLinear.

    Args:
        module: PyTorch module to modify
        num_centers: Number of RSWAF centers
        skip_names: Set of full path names to skip (e.g., {"lm_head", "token_embedding"})
        skip_module_types: Tuple of module types to skip recursing into (e.g., nn.MultiheadAttention)
        _prefix: Internal prefix for tracking full path

    Returns:
        Modified module with FasterKAN layers
    """
    if skip_names is None:
        skip_names = set()
    if skip_module_types is None:
        # Skip nn.MultiheadAttention - it directly accesses out_proj.weight
        skip_module_types = (nn.MultiheadAttention,)

    for name, child in module.named_children():
        full_name = f"{_prefix}.{name}" if _prefix else name

        # Skip certain module types entirely (they access Linear internals)
        if isinstance(child, skip_module_types):
            continue

        if isinstance(child, nn.Linear):
            # Skip certain layers (output head, embeddings)
            if name in skip_names or full_name in skip_names:
                continue
            # Replace with FasterKAN
            kan_linear = FasterKANLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                num_centers=num_centers,
            )
            setattr(module, name, kan_linear)
        else:
            # Recurse
            replace_linear_with_fasterkan(
                child,
                num_centers=num_centers,
                skip_names=skip_names,
                skip_module_types=skip_module_types,
                _prefix=full_name,
            )

    return module
