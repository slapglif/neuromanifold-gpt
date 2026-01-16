# neuromanifold_gpt/model/block.py
"""
NeuroManifold Transformer Block.

Combines: SDR -> Manifold -> Spectral -> FHN Attention -> SwiGLU

This block wires together all components:
1. SDR projection to embedding space
2. ManifoldProjection (SDR -> manifold coordinates + metric)
3. SpectralDecomposition (coords -> eigenvectors)
4. FHNAttention (attention via excitable wave dynamics)
5. SwiGLU MLP (LLaMA-style gated linear unit)
6. Pre-norm architecture (layer norms before attention/MLP)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold import ManifoldProjection
from .spectral import SpectralDecomposition
from .attention.fhn import FHNAttention
from .attention.knot import KnotAttention
from .attention.kaufmann import KaufmannAttention
from .kan.cheby import ChebyKANFFN
from .kan.wave import WaveKANFFN
from .kan.faster import FasterKANFFN
from .mhc import HyperConnections, Residual, get_init_and_expand_reduce_stream_functions
from ..config.block_config import NeuroManifoldBlockConfig


class SwiGLU(nn.Module):
    """LLaMA-style SwiGLU FFN.

    FFN(x) = (SiLU(xW_gate) ⊙ xW_up) W_down

    Uses 2/3 hidden dim to match parameter count of standard FFN.
    More expressive than GELU with similar compute.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        # LLaMA-style: 2/3 hidden dim for gate+up, then down
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class NeuroManifoldBlock(nn.Module):
    """Single transformer block with manifold-spectral-fhn attention.

    This block implements a neuromorphic transformer architecture that combines:
    - Sparse Distributed Representations (SDR) for encoding
    - Manifold-constrained transformations for geometric structure
    - Spectral decomposition for frequency-domain processing
    - FitzHugh-Nagumo (FHN) dynamics for soliton wave attention
    - KAN-based feed-forward networks for better function approximation
    - Multi-stream hyper-connections (mHC) for training stability

    Configuration is managed through a structured NeuroManifoldBlockConfig object
    that composes five specialized sub-configs:

    - **FHNConfig**: FitzHugh-Nagumo soliton wave dynamics for attention propagation
      Controls threshold, time constants, integration scheme (IMEX), and parallelization

    - **KANConfig**: Kolmogorov-Arnold Network configuration for FFN layers
      Controls basis function type (faster/wave/cheby), degree, and layer coverage

    - **MHCConfig**: Manifold-constrained hyper-connections for training stability
      Controls residual routing, Sinkhorn-Knopp iterations, and parallel stream count

    - **MLAConfig**: Multi-head latent attention for KV cache compression (optional)
      Controls latent dimension and decoupled RoPE dimension for memory efficiency

    - **MoEConfig**: Mixture of experts for conditional computation (optional)
      Controls expert count, routing strategy, and shared expert configuration

    Args:
        config: NeuroManifoldBlockConfig instance containing all block hyperparameters
                See neuromanifold_gpt.config.block_config for detailed configuration options

    Example:
        >>> from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig
        >>> config = NeuroManifoldBlockConfig(
        ...     sdr_size=2048,
        ...     embed_dim=384,
        ...     n_heads=8,
        ... )
        >>> block = NeuroManifoldBlock(config)
        >>> output, info = block(sdr_input)  # (B, T, sdr_size) -> (B, T, embed_dim)
    """

    def __init__(self, config: NeuroManifoldBlockConfig):
        super().__init__()
        self.config = config

        # SDR to embedding (skip if dimensions match for efficiency)
        if self.config.sdr_size != self.config.embed_dim:
            self.sdr_proj = nn.Linear(self.config.sdr_size, self.config.embed_dim)
        else:
            self.sdr_proj = nn.Identity()  # No projection needed in dense mode

        # Manifold + Spectral (skip if requested for speed)
        if not self.config.skip_manifold_spectral:
            self.manifold = ManifoldProjection(self.config.sdr_size, self.config.manifold_dim)
            self.spectral = SpectralDecomposition(self.config.manifold_dim, self.config.n_eigenvectors)
        else:
            self.manifold = None
            self.spectral = None

        # FHN attention (with semi-implicit IMEX scheme)
        if self.config.attention_type == "kaufmann":
            # The Full Trifecta Model
            self.attention = KaufmannAttention(
                self.config.embed_dim,
                self.config.n_heads,
                manifold_dim=self.config.manifold_dim,
                fhn_threshold=self.config.fhn.fhn_threshold,
                fhn_tau=self.config.fhn.fhn_tau,
                use_imex=self.config.fhn.use_fhn_imex,
                use_partitioning=self.config.fhn.use_fhn_partitioning,
                use_fused=self.config.fhn.use_fhn_fused
            )
        else:
            self.attention = FHNAttention(
                embed_dim=self.config.embed_dim,
                n_heads=self.config.n_heads,
                threshold=self.config.fhn.fhn_threshold,
                tau=self.config.fhn.fhn_tau,
                pulse_width_base=self.config.fhn.pulse_width_base,
                dropout=self.config.dropout,
                n_fhn_steps=self.config.fhn.n_fhn_steps,
                use_imex=self.config.fhn.use_fhn_imex,
                use_partitioning=self.config.fhn.use_fhn_partitioning,
                use_fused=self.config.fhn.use_fhn_fused
            )

        # Knot attention (optional)
        if self.config.attention_type == "knot" and self.config.attention_type != "kaufmann":
            self.knot_attention = KnotAttention(
                embed_dim=self.config.embed_dim,
                manifold_dim=self.config.manifold_dim,
                n_heads=self.config.n_heads
            )
            # Gating for combining FHN and Knot attention
            self.attn_gate = nn.Linear(self.config.embed_dim, 2)

        # FFN: SwiGLU or ChebyKAN or WaveKAN or FasterKAN
        if self.config.kan.use_kan:
            mlp_hidden = int(self.config.embed_dim * self.config.mlp_ratio)
            if self.config.kan.kan_type == "faster":
                # FasterKAN with RSWAF basis (default, fastest)
                self.mlp = FasterKANFFN(
                    self.config.embed_dim,
                    mlp_hidden,
                    num_centers=self.config.kan.kan_num_centers,
                    dropout=self.config.dropout
                )
            elif self.config.kan.kan_type == "cheby":
                # ChebyKAN FFN
                self.mlp = ChebyKANFFN(
                    self.config.embed_dim,
                    mlp_hidden,
                    degree=self.config.kan.kan_degree,
                    dropout=self.config.dropout
                )
            elif self.config.kan.kan_type == "wave":
                # WaveKAN FFN
                self.mlp = WaveKANFFN(
                    self.config.embed_dim,
                    mlp_hidden,
                    wavelet_type=self.config.kan.kan_wavelet,
                    dropout=self.config.dropout,
                    use_fast_wavekan=self.config.kan.use_fast_wavekan
                )
            else:
                raise ValueError(f"Unknown KAN type: {self.config.kan.kan_type}")
        else:
            # SwiGLU FFN (LLaMA-style, 2/3 hidden dim to match param count)
            # Standard FFN: 2 * dim * hidden = 2 * d * 4d = 8d²
            # SwiGLU: 3 * dim * hidden = 3 * d * (8/3)d = 8d² (same params)
            mlp_hidden = int(self.config.embed_dim * self.config.mlp_ratio * 2 / 3)
            self.mlp = SwiGLU(self.config.embed_dim, mlp_hidden, dropout=self.config.dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(self.config.embed_dim)
        self.norm2 = nn.LayerNorm(self.config.embed_dim)

        # mHC residual connections (from DeepSeek paper)
        # Reference: https://arxiv.org/abs/2512.24880
        # The new implementation correctly wraps sublayers with H_pre/H_res/H_post
        if self.config.mhc.use_mhc:
            if self.config.mhc.use_full_mhc and self.config.mhc.mhc_n_streams > 1:
                # Full multi-stream mHC with Sinkhorn-Knopp projection
                # Note: Multi-stream requires expand/reduce at GPT level
                self.mhc_attn = HyperConnections(
                    self.config.mhc.mhc_n_streams,
                    dim=self.config.embed_dim,
                    sinkhorn_iters=self.config.mhc.mhc_sinkhorn_iters,
                    sinkhorn_tau=self.config.mhc.mhc_sinkhorn_tau,
                    use_fused=self.config.mhc.use_mhc_fused,
                )
                self.mhc_mlp = HyperConnections(
                    self.config.mhc.mhc_n_streams,
                    dim=self.config.embed_dim,
                    sinkhorn_iters=self.config.mhc.mhc_sinkhorn_iters,
                    sinkhorn_tau=self.config.mhc.mhc_sinkhorn_tau,
                    use_fused=self.config.mhc.use_mhc_fused,
                )
            else:
                # Single-stream fallback (simple residual)
                self.mhc_attn = Residual()
                self.mhc_mlp = Residual()

    def forward(self, sdr: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            sdr: (B, T, sdr_size) or (B*S, T, sdr_size) if multi-stream mHC

        Returns:
            out: (B, T, embed_dim) or (B*S, T, embed_dim) if multi-stream
            info: diagnostic dict (includes ortho_loss for training)
        """
        # Project SDR
        x = self.sdr_proj(sdr)

        # Manifold projection and spectral decomposition (skip if requested for speed)
        if not self.config.skip_manifold_spectral:
            coords, metric = self.manifold(sdr)
            spectral_basis, spectral_freqs, ortho_loss = self.spectral(coords, metric)
        else:
            # Dummy values when skipping manifold/spectral
            coords = None
            metric = None
            spectral_basis = None
            spectral_freqs = None
            ortho_loss = torch.tensor(0.0, device=x.device)

        # FHN attention + residual with mHC
        if self.config.mhc.use_mhc:
            # New mHC architecture: H_pre computes branch input, H_post adds to residual
            branch_input, add_residual_fn = self.mhc_attn(x)
            attn_out, attn_info = self.attention(self.norm1(branch_input), spectral_basis)

            # Knot attention (if enabled)
            if self.config.attention_type == "knot":
                knot_out, knot_info = self.knot_attention(self.norm1(branch_input), coords)
                attn_info.update(knot_info)
                gate = F.softmax(self.attn_gate(branch_input), dim=-1)
                attn_out = gate[..., 0:1] * attn_out + gate[..., 1:2] * knot_out

            # Add residual via H_res + H_post
            x = add_residual_fn(attn_out)
        else:
            # Standard residual connection
            attn_out, attn_info = self.attention(self.norm1(x), spectral_basis)

            if self.config.use_knot_attention:
                knot_out, knot_info = self.knot_attention(self.norm1(x), coords)
                attn_info.update(knot_info)
                gate = F.softmax(self.attn_gate(x), dim=-1)
                attn_out = gate[..., 0:1] * attn_out + gate[..., 1:2] * knot_out

            x = x + attn_out

        # MLP + residual with mHC
        if self.config.mhc.use_mhc:
            branch_input_mlp, add_residual_fn_mlp = self.mhc_mlp(x)
            mlp_out = self.mlp(self.norm2(branch_input_mlp))
            x = add_residual_fn_mlp(mlp_out)
        else:
            mlp_out = self.mlp(self.norm2(x))
            x = x + mlp_out

        info = {
            'manifold_coords': coords,
            'metric': metric,
            'spectral_basis': spectral_basis,
            'spectral_freqs': spectral_freqs,
            'ortho_loss': ortho_loss,  # Add to training loss
            **attn_info
        }

        return x, info
