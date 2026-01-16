"""
Unified checkpoint loader for NeuroManifoldGPT.

Provides backward-compatible loading of both unified and separated checkpoint formats:
- Unified format: Single .pt/.ckpt file with model and optimizer state
- Separated format: Separate -model.pt and -optimizer.pt files

This loader automatically detects the checkpoint format and loads accordingly.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import torch
from loguru import logger


def _is_separated_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint path refers to separated format.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if this is a separated checkpoint (has corresponding -model.pt file)
    """
    # Check if the path itself is a model.pt file
    if checkpoint_path.endswith('-model.pt'):
        return True

    # Check if there's a corresponding -model.pt file
    # E.g., checkpoint-step001000.pt -> checkpoint-step001000-model.pt
    base_path = checkpoint_path.rsplit('.', 1)[0]
    model_path = f"{base_path}-model.pt"
    return os.path.exists(model_path)


def _find_separated_checkpoint_paths(checkpoint_path: str) -> Tuple[str, Optional[str]]:
    """Find model and optimizer file paths for a separated checkpoint.

    Args:
        checkpoint_path: Base checkpoint path or model.pt path

    Returns:
        Tuple of (model_path, optimizer_path)
        optimizer_path will be None if it doesn't exist

    Raises:
        FileNotFoundError: If model.pt file cannot be found
    """
    # If given a -model.pt file directly, use it
    if checkpoint_path.endswith('-model.pt'):
        model_path = checkpoint_path
        # Derive optimizer path: checkpoint-step001000-model.pt -> checkpoint-step001000-optimizer.pt
        optimizer_path = checkpoint_path.replace('-model.pt', '-optimizer.pt')
    else:
        # Given a base path or unified checkpoint path
        base_path = checkpoint_path.rsplit('.', 1)[0]
        model_path = f"{base_path}-model.pt"
        optimizer_path = f"{base_path}-optimizer.pt"

    # Verify model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Check if optimizer exists
    if not os.path.exists(optimizer_path):
        optimizer_path = None

    return model_path, optimizer_path


def load_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu',
    load_optimizer: bool = True,
    weights_only: bool = False,
) -> Dict[str, Any]:
    """Load checkpoint in unified or separated format.

    This function provides backward-compatible loading:
    - Unified format (.pt, .ckpt): Loads single file with all state
    - Separated format (-model.pt, -optimizer.pt): Loads from separate files

    The returned dictionary has a unified structure:
    {
        'model': state_dict or 'model_state_dict': state_dict,
        'config': config,
        'optimizer_states': [optimizer_state_dict, ...],  # if load_optimizer=True
        'lr_scheduler_states': [scheduler_state_dict, ...],  # if load_optimizer=True
        'global_step': int,
        'epoch': int,  # if available
        ... other keys from original checkpoint
    }

    Args:
        checkpoint_path: Path to checkpoint (unified or separated format)
        device: Device to load checkpoint onto ('cpu', 'cuda', etc.)
        load_optimizer: If True, load optimizer state (for training resumption)
        weights_only: If True, only load weights (PyTorch 2.6+ security feature)

    Returns:
        Dictionary with checkpoint data in unified format

    Raises:
        FileNotFoundError: If checkpoint file(s) not found
    """
    # Detect checkpoint format
    is_separated = _is_separated_checkpoint(checkpoint_path)

    if is_separated:
        # Load separated checkpoint format
        logger.info(f"Loading separated checkpoint from {checkpoint_path}")
        model_path, optimizer_path = _find_separated_checkpoint_paths(checkpoint_path)

        # Load model state
        logger.debug(f"Loading model from {model_path}")
        model_checkpoint = torch.load(model_path, map_location=device, weights_only=weights_only)

        # Build unified checkpoint structure
        checkpoint = {
            'model': model_checkpoint.get('model_state_dict', {}),
            'config': model_checkpoint.get('config'),
            'global_step': model_checkpoint.get('global_step', 0),
        }

        # Load optimizer state if requested and available
        if load_optimizer and optimizer_path is not None:
            logger.debug(f"Loading optimizer from {optimizer_path}")
            optimizer_checkpoint = torch.load(optimizer_path, map_location=device, weights_only=weights_only)

            checkpoint['optimizer_states'] = optimizer_checkpoint.get('optimizer_states', [])
            checkpoint['lr_scheduler_states'] = optimizer_checkpoint.get('lr_scheduler_states', [])
            checkpoint['epoch'] = optimizer_checkpoint.get('epoch', 0)

            logger.info("Loaded model and optimizer state from separated checkpoints")
        elif load_optimizer and optimizer_path is None:
            logger.warning(f"Optimizer checkpoint not found: expected at {checkpoint_path.replace('-model.pt', '-optimizer.pt')}")
            logger.warning("Continuing with model-only checkpoint")
        else:
            logger.info("Loaded model-only checkpoint (optimizer state not requested)")

    else:
        # Load unified checkpoint format (backward compatibility)
        logger.info(f"Loading unified checkpoint from {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)

        # Ensure consistent structure
        # Some old checkpoints might have 'model_state_dict' instead of 'model'
        if 'model_state_dict' in checkpoint and 'model' not in checkpoint:
            checkpoint['model'] = checkpoint['model_state_dict']

        logger.info("Loaded unified checkpoint (backward compatible format)")

    return checkpoint


def load_model_only(
    checkpoint_path: str,
    device: str = 'cpu',
    weights_only: bool = False,
) -> Dict[str, Any]:
    """Load only model weights, ignoring optimizer state.

    Convenience wrapper around load_checkpoint with load_optimizer=False.

    Args:
        checkpoint_path: Path to checkpoint (unified or separated format)
        device: Device to load checkpoint onto
        weights_only: If True, only load weights (PyTorch 2.6+ security)

    Returns:
        Dictionary with model weights and config
    """
    return load_checkpoint(
        checkpoint_path,
        device=device,
        load_optimizer=False,
        weights_only=weights_only,
    )


def get_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract model state_dict from loaded checkpoint.

    Handles both formats:
    - checkpoint['model']
    - checkpoint['model_state_dict']

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        Model state_dict

    Raises:
        KeyError: If no model state found in checkpoint
    """
    if 'model' in checkpoint:
        return checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    else:
        raise KeyError("No model state found in checkpoint (expected 'model' or 'model_state_dict' key)")


def has_optimizer_state(checkpoint: Dict[str, Any]) -> bool:
    """Check if checkpoint contains optimizer state.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        True if checkpoint has optimizer state
    """
    # Check for separated format
    if 'optimizer_states' in checkpoint:
        return len(checkpoint.get('optimizer_states', [])) > 0

    # Check for unified format (various possible keys)
    unified_keys = ['optimizer', 'optimizer_state_dict', 'optimizer_state']
    return any(key in checkpoint for key in unified_keys)
