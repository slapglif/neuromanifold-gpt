"""
GPU Detection Utility

Provides GPU capability detection and automatic attention backend selection
for optimal performance across different GPU architectures.
"""

import torch
from typing import Optional, Dict, Any


def detect_gpu_capability() -> Dict[str, Any]:
    """
    Detect GPU capabilities and architecture information.

    Returns:
        dict: GPU information including:
            - available (bool): Whether CUDA GPU is available
            - name (str): GPU device name
            - compute_capability (tuple): (major, minor) compute capability version
            - cuda_version (str): CUDA version string
            - driver_version (str): GPU driver version
            - supports_flash_attention (bool): Whether Flash Attention is supported
            - supports_xformers (bool): Whether xformers is likely supported
            - supports_triton (bool): Whether Triton kernels are likely supported
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": "CPU",
            "compute_capability": (0, 0),
            "cuda_version": "N/A",
            "driver_version": "N/A",
            "supports_flash_attention": False,
            "supports_xformers": False,
            "supports_triton": False,
        }

    device_props = torch.cuda.get_device_properties(0)
    compute_capability = (device_props.major, device_props.minor)

    # Flash Attention requires Ampere (SM 8.0) or newer
    # Ampere: RTX 30xx series, A100
    # Ada Lovelace: RTX 40xx series
    supports_flash = compute_capability[0] >= 8

    # xformers supports Volta (SM 7.0) and newer
    # Volta: V100, Titan V
    # Turing: RTX 20xx series
    supports_xformers = compute_capability[0] >= 7

    # Triton requires CUDA GPU, SM 7.0+
    supports_triton = compute_capability[0] >= 7

    return {
        "available": True,
        "name": device_props.name,
        "compute_capability": compute_capability,
        "cuda_version": torch.version.cuda or "N/A",
        "driver_version": torch.cuda.get_device_capability(0),
        "supports_flash_attention": supports_flash,
        "supports_xformers": supports_xformers,
        "supports_triton": supports_triton,
    }


def supports_flash_attention() -> bool:
    """
    Check if the current GPU supports Flash Attention.

    Flash Attention requires Ampere (SM 8.0) or newer architecture.
    This includes:
    - RTX 30xx series (3060, 3070, 3080, 3090)
    - RTX 40xx series (4060, 4070, 4080, 4090)
    - A100, A40, A30, A10 data center GPUs
    - H100 and newer

    Returns:
        bool: True if Flash Attention is supported, False otherwise
    """
    gpu_info = detect_gpu_capability()
    return gpu_info["supports_flash_attention"]


def get_optimal_attention_backend() -> str:
    """
    Determine the optimal attention backend for the current GPU.

    Selection logic:
    1. Flash Attention: Ampere+ (SM 8.0+) - fastest, most memory efficient
    2. xformers: Volta+ (SM 7.0+) - good performance, broader compatibility
    3. Triton: Volta+ (SM 7.0+) - custom kernels for specialized patterns
    4. manual: CPU or older GPUs - standard PyTorch implementation

    Returns:
        str: Recommended backend ('flash', 'xformers', 'triton', or 'manual')
    """
    gpu_info = detect_gpu_capability()

    if not gpu_info["available"]:
        return "manual"

    # Check for Flash Attention support (best option)
    if gpu_info["supports_flash_attention"]:
        # Verify PyTorch version supports scaled_dot_product_attention
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "flash"

    # Fall back to xformers for Volta+ GPUs
    if gpu_info["supports_xformers"]:
        try:
            import xformers  # noqa: F401
            return "xformers"
        except ImportError:
            pass

    # Triton as alternative for custom kernels
    if gpu_info["supports_triton"]:
        try:
            import triton  # noqa: F401
            return "triton"
        except ImportError:
            pass

    # Fall back to manual implementation
    return "manual"


def get_gpu_info_summary() -> str:
    """
    Get a human-readable summary of GPU capabilities.

    Returns:
        str: Formatted string with GPU information and recommended backend
    """
    gpu_info = detect_gpu_capability()

    if not gpu_info["available"]:
        return "No CUDA GPU available. Using CPU with manual attention backend."

    backend = get_optimal_attention_backend()

    summary = f"""GPU Information:
  Device: {gpu_info['name']}
  Compute Capability: {gpu_info['compute_capability'][0]}.{gpu_info['compute_capability'][1]}
  CUDA Version: {gpu_info['cuda_version']}

Backend Support:
  Flash Attention: {'✓' if gpu_info['supports_flash_attention'] else '✗'}
  xformers: {'✓' if gpu_info['supports_xformers'] else '✗'}
  Triton: {'✓' if gpu_info['supports_triton'] else '✗'}

Recommended Backend: {backend}"""

    return summary
