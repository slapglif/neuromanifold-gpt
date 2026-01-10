"""Configuration dataclasses for NeuroManifoldGPT.

This module defines the configuration structure for the NeuroManifoldGPT model,
including settings for:
- SDR (Sparse Distributed Representation) encoding
- Manifold projection and spectral decomposition
- Soliton attention dynamics
- Engram memory hierarchy
- DAG planning and imagination modules
"""

from dataclasses import dataclass, field


@dataclass
class NeuroManifoldConfig:
    """Main configuration for NeuroManifoldGPT model.

    This configuration covers all aspects of the novel architecture:
    - Token embedding via Semantic Folding (SDR)
    - Manifold projection with learned Riemannian metric
    - Spectral attention via eigendecomposition
    - Soliton-based attention propagation (Kaufmann-Heimburg model)
    - SDR Engram memory for infinite context
    - Forced DAG planning for System 2 thinking
    - Consistency model imagination module

    Attributes:
        vocab_size: Vocabulary size (default 50304 for GPT-2 + padding)
        block_size: Maximum sequence length (context window)
        sdr_size: Size of SDR binary vectors (bits)
        sdr_sparsity: Target sparsity ratio (~2% for biological plausibility)
        sdr_n_active: Number of active bits (computed from size * sparsity)
        sdr_embed_dim: Embedding dimension for SDR projection
        sdr_context_size: Context window for semantic folding
        manifold_dim: Dimension of the learned manifold space
        n_neighbors: Number of neighbors for manifold construction
        n_eigenvectors: Number of eigenvectors for spectral attention
        spectral_sigma: Bandwidth for spectral kernel
        n_heads: Number of attention heads
        soliton_threshold: Firing threshold for soliton activation
        soliton_tau: Time constant for soliton dynamics
        soliton_velocity: Propagation velocity of soliton waves
        pulse_width_base: Base width of soliton pulses
        n_layer: Number of transformer layers
        n_embd: Embedding dimension (must be divisible by n_heads)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
        engram_capacity: Maximum engrams in memory
        engram_threshold: SDR overlap threshold for retrieval
        l1_capacity: L1 (working memory) capacity
        l2_capacity: L2 (short-term) capacity
        l3_capacity: L3 (long-term) capacity
        max_dag_depth: Maximum depth of task DAG
        imagination_steps: Number of diffusion steps for imagination
        imagination_dim: Dimension of imagination latent space
    """

    # Vocabulary and sequence
    vocab_size: int = 50304
    block_size: int = 1024

    # SDR (Sparse Distributed Representation) configuration
    sdr_size: int = 2048
    sdr_sparsity: float = 0.02
    sdr_n_active: int = field(init=False)  # Computed in __post_init__
    sdr_embed_dim: int = 256
    sdr_context_size: int = 5

    # Manifold projection configuration
    manifold_dim: int = 64
    n_neighbors: int = 15
    n_eigenvectors: int = 32
    spectral_sigma: float = 1.0

    # Attention configuration
    n_heads: int = 8

    # Soliton dynamics (Kaufmann-Heimburg model)
    soliton_threshold: float = 0.5
    soliton_tau: float = 0.1
    soliton_velocity: float = 1.0
    pulse_width_base: int = 4

    # Model architecture
    n_layer: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False

    # Engram memory configuration
    engram_capacity: int = 1000
    engram_threshold: float = 0.3
    l1_capacity: int = 100
    l2_capacity: int = 500
    l3_capacity: int = 1000

    # DAG planning and imagination
    max_dag_depth: int = 8
    imagination_steps: int = 4
    imagination_dim: int = 256

    def __post_init__(self) -> None:
        """Validate configuration and compute derived values."""
        # Compute number of active SDR bits
        self.sdr_n_active = int(self.sdr_size * self.sdr_sparsity)

        # Validate n_embd is divisible by n_heads
        assert self.n_embd % self.n_heads == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        """Compute dimension per attention head."""
        return self.n_embd // self.n_heads


@dataclass
class NeuroManifoldConfigNano(NeuroManifoldConfig):
    """Nano preset for fast experimentation and testing.

    This is a minimal configuration suitable for:
    - Quick iteration during development
    - Unit testing
    - Resource-constrained environments
    - Debugging model components

    Compared to base config:
    - Reduced block_size (256 vs 1024)
    - Fewer layers (4 vs 6)
    - Smaller embedding (128 vs 384)
    - Fewer heads (4 vs 8)
    - Reduced manifold dimensions
    - Smaller SDR size
    """

    block_size: int = 256
    n_layer: int = 4
    n_heads: int = 4
    n_embd: int = 128
    manifold_dim: int = 32
    n_eigenvectors: int = 16
    sdr_size: int = 1024
