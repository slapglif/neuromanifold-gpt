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
