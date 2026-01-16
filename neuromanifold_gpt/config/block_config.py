"""Configuration dataclasses for NeuroManifoldBlock.

This module defines configuration structures for NeuroManifoldBlock parameters,
reducing the parameter explosion by grouping related settings into config objects:
- FHN (Fitzhugh-Nagumo) dynamics configuration
- KAN (Kolmogorov-Arnold Network) configuration
- mHC (Manifold-Constrained Hyper-Connections) configuration
- MLA (Multi-Head Latent Attention) configuration
- MoE (Mixture of Experts) configuration
"""

from dataclasses import dataclass


@dataclass
class FHNConfig:
    """Configuration for FHN (Fitzhugh-Nagumo) dynamics.

    FHN models excitable media for neural soliton wave propagation in attention.
    This is a simplified Hodgkin-Huxley model with fast-slow dynamics for
    biologically-inspired attention propagation.

    Attributes:
        fhn_threshold: Firing threshold for soliton activation (default 0.5)
        fhn_tau: Time constant for slow recovery dynamics (default 12.5)
                 Critical for slow-fast separation in FHN dynamics
        fhn_velocity: Propagation velocity of soliton waves (default 1.0)
        pulse_width_base: Base width of soliton pulses (default 4)
        n_fhn_steps: Number of integration steps (default 2 for IMEX scheme)
        use_fhn_imex: Use semi-implicit IMEX (Implicit-Explicit) scheme (default True)
                      Provides better stability than explicit Euler
        use_fhn_partitioning: Enable energy balancing for stability (default True)
        use_fhn_fused: Use fused kernel implementation (default False)
                       Disabled in favor of JIT compilation
        use_fhn_parallel: Use FFT-based Parallel Scan for linearized FHN (default True)
                          Enables maximum speed via parallel computation
    """

    fhn_threshold: float = 0.5
    fhn_tau: float = 12.5
    fhn_velocity: float = 1.0
    pulse_width_base: int = 4
    n_fhn_steps: int = 2
    use_fhn_imex: bool = True
    use_fhn_partitioning: bool = True
    use_fhn_fused: bool = False
    use_fhn_parallel: bool = True


@dataclass
class KANConfig:
    """Configuration for KAN (Kolmogorov-Arnold Network) FFN.

    KAN replaces traditional MLPs with learnable basis functions for better
    function approximation. Uses RSWAF (Rational + SiLU + Wavelet + Adaptive Fusion)
    basis for FFN layers, aligning with FHN soliton wave dynamics conceptually.

    Attributes:
        use_kan: Enable KAN for FFN layers (default True)
        kan_type: Type of KAN basis ("faster" for RSWAF, "wave" for wavelet, "cheby" for Chebyshev)
                  Default "faster" uses RSWAF for optimal speed/stability
        kan_degree: Polynomial degree for ChebyKAN (default 4)
                    Only used when kan_type="cheby"
        kan_wavelet: Wavelet type for WaveKAN (default "dog")
                     "dog" (Difference of Gaussians) is fastest and most stable
                     Only used when kan_type="wave"
        use_fast_wavekan: Use efficient WaveKAN with shared scale/translation (default True)
                          Reduces parameters significantly
        kan_num_centers: Number of RSWAF basis centers for FasterKAN (default 3)
                         Trade-off between expressivity and efficiency
        use_kan_everywhere: Replace ALL nn.Linear layers with KAN (default False)
                            When False, only FFN uses KAN; projections use Linear
                            WARNING: Enabling causes parameter explosion
    """

    use_kan: bool = True
    kan_type: str = "faster"
    kan_degree: int = 4
    kan_wavelet: str = "dog"
    use_fast_wavekan: bool = True
    kan_num_centers: int = 3
    use_kan_everywhere: bool = False


@dataclass
class MHCConfig:
    """Configuration for mHC (Manifold-Constrained Hyper-Connections).

    mHC implements DeepSeek-style hyper-connections for training stability.
    Reference: https://arxiv.org/abs/2512.24880

    Architecture: x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
    - H_res: Doubly stochastic residual matrix via Sinkhorn-Knopp (Birkhoff polytope)
    - H_pre/H_post: Routing matrices with softmax over streams for multi-stream processing

    This provides better gradient flow and training stability compared to standard
    residual connections, especially for deep networks.

    Attributes:
        use_mhc: Enable mHC for training stability (default True)
        use_full_mhc: Use full multi-stream mHC vs simplified version (default True)
                      Full version has parallel streams with learned routing
        mhc_n_streams: Number of parallel streams for full mHC (default 2)
                       2 streams provide good efficiency/expressivity trade-off
        mhc_residual_weight: Initial identity mapping bias (default 0.9)
                             Higher values bias toward identity at initialization
        mhc_sinkhorn_iters: Sinkhorn-Knopp iterations for doubly stochastic projection (default 5)
                            3-5 iterations are sufficient for convergence
        mhc_sinkhorn_tau: Sinkhorn temperature for smoothness (default 0.05)
                          Lower values make distribution sharper
    """

    use_mhc: bool = True
    use_full_mhc: bool = True
    mhc_n_streams: int = 2
    mhc_residual_weight: float = 0.9
    mhc_sinkhorn_iters: int = 5
    mhc_sinkhorn_tau: float = 0.05


@dataclass
class MLAConfig:
    """Configuration for MLA (Multi-Head Latent Attention).

    MLA implements DeepSeek-style KV compression for memory efficiency.
    Compresses key-value cache to low-dimensional latent space, achieving
    ~8x memory reduction compared to standard multi-head attention.

    This is particularly useful for long-context scenarios where KV cache
    memory becomes a bottleneck. The latent compression is learned jointly
    with the attention mechanism.

    Key innovation: Decoupled RoPE (Rotary Position Embedding) dimension
    allows separate handling of positional information from content.

    Attributes:
        use_mla: Enable MLA for KV compression (default False)
                 Off by default as it adds architectural complexity
        mla_latent_dim: KV compression dimension (default 64)
                        Lower values = more compression but less capacity
                        Typically 1/4 to 1/8 of the embedding dimension
        mla_rope_dim: Decoupled RoPE dimension (default 32)
                      Dimension for rotary position embeddings
                      Usually 1/2 of latent_dim for efficiency
    """

    use_mla: bool = False
    mla_latent_dim: int = 64
    mla_rope_dim: int = 32


@dataclass
class MoEConfig:
    """Configuration for MoE (Mixture of Experts).

    MoE implements DeepSeek-style auxiliary-loss-free Mixture of Experts.
    Uses bias-based load balancing instead of auxiliary loss terms for
    better training stability and performance.

    Traditional MoE uses auxiliary losses to encourage load balancing,
    which can hurt model quality. DeepSeek's approach uses learned biases
    for each expert, avoiding this trade-off.

    Architecture includes optional shared expert (always active) and
    E7 curriculum-based routing for progressive learning.

    Attributes:
        use_moe: Enable MoE for conditional computation (default False)
                 Off by default as it significantly increases parameters
        moe_n_experts: Total number of experts (default 8)
                       More experts = more capacity but higher memory cost
                       Typical range: 8-64 experts
        moe_n_active: Number of active experts per token (default 2)
                      Trade-off between computation and expressivity
                      Typical: 2 for efficiency, 4-8 for quality
        use_shared_expert: Always-active shared expert (default True)
                          DeepSeek-style shared expert captures common patterns
                          Improves stability and reduces expert collapse
        use_e7_routing: Route by E7 curriculum tier (default False)
                        Uses E7 lattice structure for progressive routing
                        Experimental: curriculum-based expert specialization
    """

    use_moe: bool = False
    moe_n_experts: int = 8
    moe_n_active: int = 2
    use_shared_expert: bool = True
    use_e7_routing: bool = False
