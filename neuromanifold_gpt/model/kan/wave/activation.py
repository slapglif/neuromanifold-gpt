import torch


@torch.jit.script
def wavekan_activation(x_scaled: torch.Tensor, wavelet_type: str) -> torch.Tensor:
    """
    JIT-compiled wavelet activation function.
    """
    if wavelet_type == "mexican_hat":
        # C * (1 - t^2) * exp(-t^2/2)
        # C = 0.8673250705840776
        t2 = x_scaled * x_scaled
        term1 = 1.0 - t2
        term2 = torch.exp(-0.5 * t2)
        wavelet = 0.8673250705840776 * term1 * term2
    elif wavelet_type == "morlet":
        # cos(5t) * exp(-t^2/2)
        t2 = x_scaled * x_scaled
        real = torch.cos(5.0 * x_scaled)
        envelope = torch.exp(-0.5 * t2)
        wavelet = envelope * real
    elif wavelet_type == "dog":
        # -t * exp(-t^2/2)
        t2 = x_scaled * x_scaled
        wavelet = -x_scaled * torch.exp(-0.5 * t2)
    else:
        # Default/Fallback (should not happen if checked before)
        wavelet = x_scaled  # Identity placeholder

    return wavelet
