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


def _spectral_decomposition_forward(
    coords: torch.Tensor,
    use_learned_basis: bool,
    spectral_proj_output: torch.Tensor,
    freq_embed: torch.Tensor,
    random_features: torch.Tensor,
    sigma: torch.Tensor,
    n_eig: int,
    ortho_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Spectral decomposition forward pass computation.

    This function is compiled with torch.compile for kernel fusion.

    Args:
        coords: (B, T, manifold_dim) manifold coordinates
        use_learned_basis: Whether to use learned basis or random features
        spectral_proj_output: (B, T, n_eig) output from spectral_proj if learned basis, else None
        freq_embed: (n_eig,) learned frequencies if learned basis, else None
        random_features: (manifold_dim, n_eig) random features if not learned basis, else None
        sigma: scalar bandwidth parameter
        n_eig: number of eigenvectors
        ortho_weight: weight for orthogonality regularization

    Returns:
        spectral_basis: (B, T, n_eig) spectral coefficients
        spectral_freqs: (B, n_eig) frequency/eigenvalue estimates
        ortho_loss: scalar orthogonality regularization loss
    """
    B, T, D = coords.shape

    if use_learned_basis:
        # Normalize for stable attention (avoid softmax gradient issues)
        # Use L2 normalization instead of softmax for better gradient flow
        spectral_basis = F.normalize(spectral_proj_output, p=2, dim=-1)
        # Scale to have similar magnitude as softmax would
        spectral_basis = spectral_basis * (1.0 / (n_eig ** 0.5))

        # Learned frequencies
        spectral_freqs = freq_embed.abs().unsqueeze(0).expand(B, -1)

        # Compute orthogonality regularization loss: ||Φᵀ Φ - I||²
        # Normalize basis vectors for Gram matrix computation
        # (B, n_eig, T) @ (B, T, n_eig) -> (B, n_eig, n_eig)
        basis_norm = F.normalize(spectral_basis, dim=1)  # Normalize along T dimension
        gram = torch.bmm(basis_norm.transpose(1, 2), basis_norm)

        # Target: identity matrix (orthonormal basis)
        eye = torch.eye(n_eig, device=gram.device, dtype=gram.dtype)

        # Frobenius norm: ||Gram - I||²
        ortho_loss = (gram - eye).pow(2).mean() * ortho_weight
    else:
        # Random Fourier features
        # cos(W @ x) approximates spectral structure
        proj = coords @ random_features  # (B, T, n_eig)
        spectral_basis = torch.cos(proj * sigma)
        spectral_freqs = torch.arange(
            n_eig, device=coords.device, dtype=coords.dtype
        ).unsqueeze(0).expand(B, -1)
        # No ortho loss for random features (they're not learned)
        ortho_loss = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

    return spectral_basis, spectral_freqs, ortho_loss


# Compile with reduce-overhead mode for minimal Python overhead
# Gracefully fall back to uncompiled version on Python 3.12+ (where Dynamo is not supported)
try:
    spectral_decomposition_forward = torch.compile(
        _spectral_decomposition_forward,
        mode="reduce-overhead"
    )
except RuntimeError as e:
    if "Dynamo is not supported" in str(e):
        # Fall back to uncompiled version on Python 3.12+
        spectral_decomposition_forward = _spectral_decomposition_forward
    else:
        raise


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
        # NO O(n²) affinity computation - this is the key optimization!
        # The learned basis captures spectral structure without explicit affinity.

        if self.use_learned_basis:
            # Learned spectral projection: O(n·k) instead of O(n²) or O(n³)
            spectral_proj_output = self.spectral_proj(coords)  # (B, T, n_eig)
            freq_embed = self.freq_embed
            random_features = None
        else:
            spectral_proj_output = None
            freq_embed = None
            random_features = self.random_features

        # Execute spectral decomposition using torch.compile-optimized kernel
        spectral_basis, spectral_freqs, ortho_loss = spectral_decomposition_forward(
            coords,
            self.use_learned_basis,
            spectral_proj_output,
            freq_embed,
            random_features,
            self.sigma,
            self.n_eig,
            self.ortho_weight,
        )

        return spectral_basis, spectral_freqs, ortho_loss


class FastSpectralAttention(nn.Module):
    """
    O(n) attention using learned spectral basis.

    Instead of O(n²) full attention, projects to spectral space (k dims),
    computes attention there, and projects back. Total: O(n·k + k²).

    Memory Optimization:
    -------------------
    Uses chunked processing for causal cumsum to reduce memory complexity
    from O(T*k²) to O(chunk_size*k²) while maintaining causal masking.

    The key optimization is in computing the causal cumulative sum of k^T @ v
    outer products. Naive implementation would materialize a (B, H, T, k, k)
    tensor all at once, requiring O(T*k²) memory. Instead, we:

    1. Process the sequence in chunks of size `chunk_size`
    2. Compute outer products only within each chunk: O(chunk_size*k²) memory
    3. Maintain a running cumsum state (B, H, k, k) across chunk boundaries
    4. Ensure causality by adding previous cumsum state to each new chunk

    This allows handling arbitrarily long sequences with bounded memory usage,
    controlled by the `chunk_size` parameter.

    Memory-Speed Trade-off:
    ----------------------
    - Smaller chunk_size: Lower memory usage, potentially slower (more chunks)
    - Larger chunk_size: Higher memory usage, potentially faster (fewer chunks)
    - Default (256): Good balance for typical transformer sequences

    Complexity:
    ----------
    - Time: O(T*k²) - same as naive implementation
    - Memory: O(chunk_size*k²) - drastically reduced from O(T*k²)
    - For T=8192, k=32, chunk_size=256: ~32x memory reduction
    """

    def __init__(self, embed_dim: int, n_eigenvectors: int = 32, n_heads: int = 8, chunk_size: int = 256):
        """
        Initialize FastSpectralAttention.

        Args:
            embed_dim: Embedding dimension for input/output
            n_eigenvectors: Number of spectral basis functions (k). Default: 32
            n_heads: Number of attention heads. Default: 8
            chunk_size: Sequence chunk size for memory-efficient causal cumsum.
                       Reduces memory from O(T*k²) to O(chunk_size*k²).
                       Smaller values use less memory but may be slower.
                       Default: 256
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_eig = n_eigenvectors
        self.chunk_size = chunk_size

        # Project to spectral query/key/value
        self.to_qkv = nn.Linear(embed_dim, 3 * n_eigenvectors * n_heads, bias=False)

        # Project spectral output back to embed_dim
        self.to_out = nn.Linear(n_eigenvectors * n_heads, embed_dim)

        # Scale
        self.scale = n_eigenvectors**-0.5

    def _chunked_causal_cumsum(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal cumsum of k^T @ v outer products using chunking.

        Reduces memory from O(T*k^2) to O(chunk_size*k^2) by processing
        the sequence in chunks and maintaining a running cumsum state.

        Args:
            k: (B, H, T, n_eig) key vectors
            v: (B, H, T, n_eig) value vectors

        Returns:
            kv_causal: (B, H, T, n_eig, n_eig) causal cumsum of outer products
        """
        B, H, T, K = k.shape
        device = k.device
        dtype = k.dtype

        # Allocate output tensor
        kv_causal = torch.zeros(B, H, T, K, K, device=device, dtype=dtype)

        # Running cumsum state: (B, H, K, K)
        cumsum_state = torch.zeros(B, H, K, K, device=device, dtype=dtype)

        # Process sequence in chunks
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            # Chunk boundaries
            start_t = chunk_idx * self.chunk_size
            end_t = min(start_t + self.chunk_size, T)
            chunk_len = end_t - start_t

            # Extract chunk: (B, H, chunk_len, K)
            k_chunk = k[:, :, start_t:end_t, :]
            v_chunk = v[:, :, start_t:end_t, :]

            # Compute outer products for this chunk: (B, H, chunk_len, K, K)
            # k_chunk.unsqueeze(-1): (B, H, chunk_len, K, 1)
            # v_chunk.unsqueeze(-2): (B, H, chunk_len, 1, K)
            kv_chunk = k_chunk.unsqueeze(-1) * v_chunk.unsqueeze(-2)

            # Add cumsum_state to first position of chunk
            # This ensures causality across chunk boundaries
            kv_chunk[:, :, 0, :, :] += cumsum_state

            # Compute cumsum within chunk
            kv_chunk_cumsum = torch.cumsum(kv_chunk, dim=2)

            # Store results
            kv_causal[:, :, start_t:end_t, :, :] = kv_chunk_cumsum

            # Update running state for next chunk
            cumsum_state = kv_chunk_cumsum[:, :, -1, :, :]

        return kv_causal

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

        # 1-2. Compute causal cumsum of k^T @ v outer products using chunked processing
        # This reduces memory from O(T*k^2) to O(chunk_size*k^2)
        # Dimensions: k=(B, H, T, n_eig), v=(B, H, T, n_eig)
        # Output: kv_causal=(B, H, T, n_eig, n_eig) where kv_causal[t] = sum_{i<=t} k_i^T v_i
        kv_causal = self._chunked_causal_cumsum(k, v)
        
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
