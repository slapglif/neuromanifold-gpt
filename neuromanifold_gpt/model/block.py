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
    """Single transformer block with manifold-spectral-fhn attention."""

    def __init__(
        self,
        sdr_size: int,
        embed_dim: int,
        manifold_dim: int = 64,
        n_eigenvectors: int = 32,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # FHN dynamics parameters
        fhn_threshold: float = 0.5,
        fhn_tau: float = 12.5,  # Fixed: proper slow-fast separation
        pulse_width_base: int = 4,
        n_fhn_steps: int = 2,  # IMEX allows 2 steps (was 5)
        use_fhn_imex: bool = True,  # Use semi-implicit scheme
        use_fhn_partitioning: bool = True,  # Enable energy balancing for stability
        use_fhn_fused: bool = True, # Enable Fused Triton Kernel
        # Knot attention
        use_knot_attention: bool = False,
        # Kaufmann Trifecta Attention
        use_kaufmann_attention: bool = False,
        # mHC (Manifold-Constrained Hyper-Connections)
        use_mhc: bool = True,  # Enable mHC by default for stability
        use_full_mhc: bool = True,  # Use full multi-stream mHC (vs simplified)
        mhc_n_streams: int = 4,  # Number of streams for full mHC
        mhc_residual_weight: float = 0.9,  # Initial identity bias
        mhc_sinkhorn_iters: int = 10,  # Sinkhorn-Knopp iterations
        mhc_sinkhorn_tau: float = 0.05,  # Sinkhorn temperature
        # KAN configuration
        use_kan: bool = True,  # Use KAN instead of SwiGLU
        kan_type: str = "faster",  # "faster", "cheby", or "wave"
        kan_degree: int = 4,  # For ChebyKAN
        kan_wavelet: str = "mexican_hat", # For WaveKAN
        use_fast_wavekan: bool = True, # For WaveKAN
        kan_num_centers: int = 8,  # For FasterKAN RSWAF centers
        # Speed optimization
        skip_manifold_spectral: bool = False,  # Skip manifold/spectral for faster training
        # MLA (Multi-Head Latent Attention) - DeepSeek style
        use_mla: bool = False,
        mla_latent_dim: int = 64,
        mla_rope_dim: int = 32,
        # MoE (Mixture of Experts) - DeepSeek style
        use_moe: bool = False,
        moe_n_experts: int = 8,
        moe_n_active: int = 2,
        use_shared_expert: bool = True,
        use_e7_routing: bool = False,
    ):
        super().__init__()

        # SDR to embedding (skip if dimensions match for efficiency)
        if sdr_size != embed_dim:
            self.sdr_proj = nn.Linear(sdr_size, embed_dim)
        else:
            self.sdr_proj = nn.Identity()  # No projection needed in dense mode
        self.skip_manifold_spectral = skip_manifold_spectral

        # Manifold + Spectral (skip if requested for speed)
        if not skip_manifold_spectral:
            self.manifold = ManifoldProjection(sdr_size, manifold_dim)
            self.spectral = SpectralDecomposition(manifold_dim, n_eigenvectors)
        else:
            self.manifold = None
            self.spectral = None

        # FHN attention (with semi-implicit IMEX scheme)
        if use_kaufmann_attention:
            # The Full Trifecta Model
            self.attention = KaufmannAttention(embed_dim, n_heads, config=None) # We need to pass config or params
            # Actually KaufmannAttention needs config for FHN params.
            # We should pass individual params or a config object.
            # Let's refactor KaufmannAttention to take params directly or construct config proxy.
            # For now, assuming we handle this in __init__ logic below
            pass
        else:
            self.attention = FHNAttention(
                embed_dim=embed_dim,
                n_heads=n_heads,
                threshold=fhn_threshold,
                tau=fhn_tau,
                pulse_width_base=pulse_width_base,
                dropout=dropout,
                n_fhn_steps=n_fhn_steps,
                use_imex=use_fhn_imex,
                use_partitioning=use_fhn_partitioning,
                use_fused=use_fhn_fused
            )

        # Knot attention (optional)
        self.use_knot_attention = use_knot_attention
        self.use_kaufmann_attention = use_kaufmann_attention
        
        if self.use_kaufmann_attention:
             # Re-init attention with params
             # We construct it using the params passed to block __init__
             self.attention = KaufmannAttention(
                 embed_dim, n_heads, 
                 manifold_dim=manifold_dim,
                 fhn_threshold=fhn_threshold,
                 fhn_tau=fhn_tau,
                 use_imex=use_fhn_imex,
                 use_partitioning=use_fhn_partitioning,
                 use_fused=use_fhn_fused
             )

        if self.use_knot_attention and not self.use_kaufmann_attention:
            self.knot_attention = KnotAttention(
                embed_dim=embed_dim,
                manifold_dim=manifold_dim,
                n_heads=n_heads
            )
            # Gating for combining FHN and Knot attention
            self.attn_gate = nn.Linear(embed_dim, 2)

        # FFN: SwiGLU or ChebyKAN or WaveKAN or FasterKAN
        if use_kan:
            mlp_hidden = int(embed_dim * mlp_ratio)
            if kan_type == "faster":
                # FasterKAN with RSWAF basis (default, fastest)
                self.mlp = FasterKANFFN(
                    embed_dim,
                    mlp_hidden,
                    num_centers=kan_num_centers,
                    dropout=dropout
                )
            elif kan_type == "cheby":
                # ChebyKAN FFN
                self.mlp = ChebyKANFFN(embed_dim, mlp_hidden, degree=kan_degree, dropout=dropout)
            elif kan_type == "wave":
                # WaveKAN FFN
                self.mlp = WaveKANFFN(
                    embed_dim,
                    mlp_hidden,
                    wavelet_type=kan_wavelet,
                    dropout=dropout,
                    use_fast_wavekan=use_fast_wavekan
                )
            else:
                raise ValueError(f"Unknown KAN type: {kan_type}")
        else:
            # SwiGLU FFN (LLaMA-style, 2/3 hidden dim to match param count)
            # Standard FFN: 2 * dim * hidden = 2 * d * 4d = 8d²
            # SwiGLU: 3 * dim * hidden = 3 * d * (8/3)d = 8d² (same params)
            mlp_hidden = int(embed_dim * mlp_ratio * 2 / 3)
            self.mlp = SwiGLU(embed_dim, mlp_hidden, dropout=dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # mHC residual connections (from DeepSeek paper)
        # Reference: https://arxiv.org/abs/2512.24880
        # The new implementation correctly wraps sublayers with H_pre/H_res/H_post
        self.use_mhc = use_mhc
        self.use_full_mhc = use_full_mhc
        self.mhc_n_streams = mhc_n_streams
        if self.use_mhc:
            if use_full_mhc and mhc_n_streams > 1:
                # Full multi-stream mHC with Sinkhorn-Knopp projection
                # Note: Multi-stream requires expand/reduce at GPT level
                self.mhc_attn = HyperConnections(
                    mhc_n_streams,
                    dim=embed_dim,
                    sinkhorn_iters=mhc_sinkhorn_iters,
                    sinkhorn_tau=mhc_sinkhorn_tau,
                )
                self.mhc_mlp = HyperConnections(
                    mhc_n_streams,
                    dim=embed_dim,
                    sinkhorn_iters=mhc_sinkhorn_iters,
                    sinkhorn_tau=mhc_sinkhorn_tau,
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
        if not self.skip_manifold_spectral:
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
        if self.use_mhc:
            # New mHC architecture: H_pre computes branch input, H_post adds to residual
            branch_input, add_residual_fn = self.mhc_attn(x)
            attn_out, attn_info = self.attention(self.norm1(branch_input), spectral_basis)

            # Knot attention (if enabled)
            if self.use_knot_attention:
                knot_out, knot_info = self.knot_attention(self.norm1(branch_input), coords)
                attn_info.update(knot_info)
                gate = F.softmax(self.attn_gate(branch_input), dim=-1)
                attn_out = gate[..., 0:1] * attn_out + gate[..., 1:2] * knot_out

            # Add residual via H_res + H_post
            x = add_residual_fn(attn_out)
        else:
            # Standard residual connection
            attn_out, attn_info = self.attention(self.norm1(x), spectral_basis)

            if self.use_knot_attention:
                knot_out, knot_info = self.knot_attention(self.norm1(x), coords)
                attn_info.update(knot_info)
                gate = F.softmax(self.attn_gate(x), dim=-1)
                attn_out = gate[..., 0:1] * attn_out + gate[..., 1:2] * knot_out

            x = x + attn_out

        # MLP + residual with mHC
        if self.use_mhc:
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
