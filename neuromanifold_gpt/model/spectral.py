# neuromanifold_gpt/model/spectral.py
"""
Spectral Decomposition for O(n) Attention.

Uses learned spectral basis instead of eigendecomposition for efficiency.
The learned basis approximates graph Laplacian eigenvectors without O(n³) cost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


class SpectralDecomposition(nn.Module):
    """
    Efficient spectral decomposition using learned basis.

    Instead of computing eigendecomposition (O(n³)), we learn a spectral
    projection that captures similar structure in O(n·k) time.

    The learned basis approximates the behavior of graph Laplacian
    eigenvectors while being differentiable and numerically stable.

    Orthogonality regularization encourages Φᵀ Φ ≈ I in spectral dimension,
    making the learned basis behave more like actual eigenvectors.
    """

    def __init__(
        self,
        manifold_dim: int,
        n_eigenvectors: int = 32,
        sigma: float = 1.0,
        use_learned_basis: bool = True,
        ortho_weight: float = 0.01,  # Weight for orthogonality regularization
    ):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.n_eig = n_eigenvectors
        self.use_learned_basis = use_learned_basis
        self.ortho_weight = ortho_weight

        # Learned spectral bandwidth
        self.log_sigma = nn.Parameter(torch.tensor(sigma).log())

        if use_learned_basis:
            # Learned spectral basis: projects coords to spectral space
            # This approximates the graph Laplacian eigenvectors
            self.spectral_proj = nn.Sequential(
                nn.Linear(manifold_dim, manifold_dim * 2),
                nn.SiLU(),
                nn.Linear(manifold_dim * 2, n_eigenvectors),
            )
            # Learned frequencies (analogous to eigenvalues)
            self.freq_embed = nn.Parameter(torch.randn(n_eigenvectors) * 0.02)
        else:
            # Random Fourier features for spectral approximation
            # These approximate the spectral structure without learning
            self.register_buffer(
                "random_features",
                torch.randn(manifold_dim, n_eigenvectors) / manifold_dim**0.5,
            )

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def compute_ortho_loss(self, spectral_basis: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality regularization loss: ||Φᵀ Φ - I||².

        Encourages the learned spectral basis to behave like actual eigenvectors
        by making columns approximately orthonormal.

        Args:
            spectral_basis: (B, T, n_eig) spectral coefficients

        Returns:
            ortho_loss: scalar loss encouraging orthogonality
        """
        # Normalize basis vectors for Gram matrix computation
        # (B, n_eig, T) @ (B, T, n_eig) -> (B, n_eig, n_eig)
        basis_norm = F.normalize(spectral_basis, dim=1)  # Normalize along T dimension
        gram = torch.bmm(basis_norm.transpose(1, 2), basis_norm)

        # Target: identity matrix (orthonormal basis)
        eye = torch.eye(self.n_eig, device=gram.device, dtype=gram.dtype)

        # Frobenius norm: ||Gram - I||²
        ortho_loss = (gram - eye).pow(2).mean()

        return ortho_loss * self.ortho_weight

    def forward(
        self,
        coords: torch.Tensor,
        metric: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute efficient spectral decomposition. O(n·k) complexity.

        Args:
            coords: (B, T, manifold_dim) manifold coordinates
            metric: optional (B, T, D, D) Riemannian metric (unused, for API compat)

        Returns:
            spectral_basis: (B, T, n_eig) spectral coefficients
            spectral_freqs: (B, n_eig) frequency/eigenvalue estimates
            ortho_loss: scalar orthogonality regularization loss
        """
        B, T, D = coords.shape

        # NO O(n²) affinity computation - this is the key optimization!
        # The learned basis captures spectral structure without explicit affinity.

        if self.use_learned_basis:
            # Learned spectral projection: O(n·k) instead of O(n²) or O(n³)
            spectral_basis = self.spectral_proj(coords)  # (B, T, n_eig)

            # Normalize for stable attention (avoid softmax gradient issues)
            # Use L2 normalization instead of softmax for better gradient flow
            spectral_basis = F.normalize(spectral_basis, p=2, dim=-1)
            # Scale to have similar magnitude as softmax would
            spectral_basis = spectral_basis * (1.0 / (self.n_eig ** 0.5))

            # Learned frequencies
            spectral_freqs = self.freq_embed.abs().unsqueeze(0).expand(B, -1)

            # Compute orthogonality regularization loss
            ortho_loss = self.compute_ortho_loss(spectral_basis)
        else:
            # Random Fourier features
            # cos(W @ x) approximates spectral structure
            proj = coords @ self.random_features  # (B, T, n_eig)
            spectral_basis = torch.cos(proj * self.sigma)
            spectral_freqs = torch.arange(
                self.n_eig, device=coords.device, dtype=coords.dtype
            ).unsqueeze(0).expand(B, -1)
            # No ortho loss for random features (they're not learned)
            ortho_loss = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

        return spectral_basis, spectral_freqs, ortho_loss


class FastSpectralAttention(nn.Module):
    """
    O(n) attention using learned spectral basis.

    Instead of O(n²) full attention, projects to spectral space (k dims),
    computes attention there, and projects back. Total: O(n·k + k²).
    """

    def __init__(self, embed_dim: int, n_eigenvectors: int = 32, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_eig = n_eigenvectors

        # Project to spectral query/key/value
        self.to_qkv = nn.Linear(embed_dim, 3 * n_eigenvectors * n_heads, bias=False)

        # Project spectral output back to embed_dim
        self.to_out = nn.Linear(n_eigenvectors * n_heads, embed_dim)

        # Scale
        self.scale = n_eigenvectors**-0.5

    def forward(
        self,
        x: torch.Tensor,
        spectral_basis: torch.Tensor,
        spectral_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spectral attention.

        Args:
            x: (B, T, embed_dim) input embeddings
            spectral_basis: (B, T, n_eig) from SpectralDecomposition
            spectral_freqs: (B, n_eig) frequency information

        Returns:
            out: (B, T, embed_dim) attended output
        """
        B, T, C = x.shape

        # Project to spectral QKV
        qkv = self.to_qkv(x)  # (B, T, 3 * n_eig * heads)
        qkv = rearrange(qkv, "b t (three h k) -> three b h t k", three=3, h=self.n_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, heads, T, n_eig)

        # Modulate by spectral basis
        q = q * spectral_basis.unsqueeze(1)  # (B, heads, T, n_eig)
        k = k * spectral_basis.unsqueeze(1)

        # Spectral attention: O(n·k + k²)
        # CAUSAL IMPLEMENTATION for autoregressive modeling
        
        # 1. Compute outer product k^T @ v in spectral space for each time step
        # Dimensions: k=(B, H, T, n_eig), v=(B, H, T, n_eig)
        # We need sum_{i<=t} k_i^T v_i
        
        # k.unsqueeze(-1): (B, H, T, n_eig, 1)
        # v.unsqueeze(-2): (B, H, T, 1, n_eig)
        # kv_prod: (B, H, T, n_eig, n_eig)
        kv_prod = k.unsqueeze(-1) * v.unsqueeze(-2)
        
        # 2. Causal accumulation (prefix sum)
        kv_causal = torch.cumsum(kv_prod, dim=2)  # (B, H, T, n_eig, n_eig)
        
        # 3. Apply query
        # q: (B, H, T, n_eig) -> (B, H, T, 1, n_eig)
        # attn_out = q @ kv_causal
        # (B, H, T, 1, n_eig) @ (B, H, T, n_eig, n_eig) -> (B, H, T, 1, n_eig)
        attn_out = torch.matmul(q.unsqueeze(-2), kv_causal).squeeze(-2)
        
        # Scale
        attn_out = attn_out * self.scale

        # Reshape and project
        attn_out = rearrange(attn_out, "b h t k -> b t (h k)")
        out = self.to_out(attn_out)

        return out
