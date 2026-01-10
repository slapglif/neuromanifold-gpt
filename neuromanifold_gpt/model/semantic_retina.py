"""Semantic Retina - Topographic feature map for Semantic Folding.

The Semantic Retina maps semantic space onto a 2D grid (like visual cortex).
Each position represents a semantic feature. Gaussian smoothing spreads 
activation locally, creating topographic organization where similar concepts
cluster together spatially.

Key concepts:
- Grid maps n_features to 2D space (like retinotopic mapping in V1)
- Gaussian smoothing creates local feature neighborhoods
- Preserves shape: (B, T, H, W) -> (B, T, H, W)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SemanticRetina(nn.Module):
    """Topographic feature map with Gaussian smoothing.
    
    Maps semantic features onto a 2D grid and applies Gaussian smoothing
    to create local feature neighborhoods. This mimics how the visual
    cortex organizes information topographically.
    
    Args:
        grid_size: Size of the 2D grid (grid_size x grid_size)
        n_features: Total number of semantic features (should equal grid_size^2)
        kernel_size: Size of the Gaussian smoothing kernel (must be odd)
        sigma: Standard deviation of the Gaussian kernel
    """
    
    def __init__(
        self,
        grid_size: int = 64,
        n_features: int = 4096,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Compute grid dimensions
        self.grid_h = int(math.sqrt(n_features))
        self.grid_w = n_features // self.grid_h
        
        # Create and register Gaussian kernel as buffer (not a parameter)
        kernel = self._make_gaussian_kernel(kernel_size, sigma)
        self.register_buffer("gaussian_kernel", kernel)
        
        # Padding to preserve spatial dimensions
        self.padding = kernel_size // 2
    
    def _make_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a normalized 2D Gaussian kernel.
        
        Args:
            kernel_size: Size of the kernel (must be odd for symmetric padding)
            sigma: Standard deviation of the Gaussian
            
        Returns:
            Normalized 2D Gaussian kernel tensor of shape (kernel_size, kernel_size)
        """
        # Create 1D coordinate grid centered at 0
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        
        # 1D Gaussian
        gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        
        # 2D Gaussian via outer product
        gauss_2d = gauss_1d.outer(gauss_1d)
        
        # Normalize to sum to 1
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        return gauss_2d
    
    def forward(self, activation_map: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to the activation map.
        
        Args:
            activation_map: Input tensor of shape (B, T, H, W)
                B = batch size
                T = sequence length (or number of channels)
                H, W = spatial dimensions of the grid
                
        Returns:
            Smoothed activation map of the same shape (B, T, H, W)
        """
        B, T, H, W = activation_map.shape
        
        # Reshape to (B*T, 1, H, W) for conv2d
        # Using einops for clarity
        x = rearrange(activation_map, "b t h w -> (b t) 1 h w")
        
        # Prepare kernel for conv2d: (out_channels=1, in_channels=1, kH, kW)
        kernel = self.gaussian_kernel.unsqueeze(0).unsqueeze(0)
        
        # Apply Gaussian smoothing via convolution
        smoothed = F.conv2d(x, kernel, padding=self.padding)
        
        # Reshape back to (B, T, H, W)
        smoothed = rearrange(smoothed, "(b t) 1 h w -> b t h w", b=B, t=T)
        
        return smoothed
    
    def extra_repr(self) -> str:
        """String representation for print(module)."""
        return (
            f"grid_size={self.grid_size}, "
            f"n_features={self.n_features}, "
            f"kernel_size={self.kernel_size}, "
            f"sigma={self.sigma}"
        )
