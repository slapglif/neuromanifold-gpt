"""Architecture search space definition for NAS.

This module defines the search space for Neural Architecture Search (NAS) on
NeuroManifoldGPT. It provides:
- ArchitectureConfig: Configuration dataclass for a specific architecture
- SearchSpace: Search space definition with valid ranges and choices

The search space covers key architectural decisions including:
- Model depth (n_layer) and width (n_embd)
- Attention mechanism configuration (type, heads, dimensions)
- Component choices (FHN, MHC, MLA, MoE, KAN variants)
- Hyperparameters (dropout, thresholds, time constants)
- Manifold and spectral decomposition parameters
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ArchitectureConfig:
    """Configuration for a specific neural architecture.

    This dataclass defines a complete architecture configuration that can be
    instantiated and trained. It captures all architectural choices including
    model size, component selection, and hyperparameters.

    The configuration is designed to be:
    - Serializable: Can be saved/loaded from JSON/dict
    - Validatable: Can check if configuration is valid
    - Convertible: Can convert to NeuroManifoldConfig for training

    Attributes:
        # Model architecture
        n_layer: Number of transformer layers (depth)
        n_embd: Embedding dimension (width)
        n_heads: Number of attention heads (must divide n_embd)

        # Attention configuration
        attention_type: Type of attention mechanism ("fhn", "kaufmann", "knot")
        use_qk_norm: Whether to use QK normalization (stability)

        # Component choices
        use_mhc: Enable Manifold-Constrained Hyper-Connections
        use_mla: Enable Multi-Head Latent Attention (KV compression)
        use_moe: Enable Mixture of Experts
        use_kan: Enable Kolmogorov-Arnold Networks for FFN

        # KAN configuration
        kan_type: Type of KAN basis ("faster", "wave", "cheby")
        kan_num_centers: Number of RSWAF basis centers (for faster KAN)

        # FHN (Fitzhugh-Nagumo) dynamics
        fhn_threshold: Firing threshold for soliton activation
        fhn_tau: Time constant for slow recovery dynamics
        use_fhn_parallel: Use FFT-based parallel scan for speed

        # Manifold projection
        manifold_dim: Dimension of the learned manifold space
        n_eigenvectors: Number of eigenvectors for spectral attention
        use_multiscale_manifold: Enable multi-scale manifold hierarchy

        # Regularization
        dropout: Dropout probability

        # Memory and SDR configuration
        use_sdr: Enable Sparse Distributed Representations
        sdr_size: Size of the SDR representation
        sdr_sparsity: Sparsity ratio for SDR (fraction of active bits)
        engram_capacity: Maximum capacity of engram memory
        engram_threshold: Threshold for engram activation

        # Metadata
        architecture_id: Unique identifier for this architecture (optional)
        search_iteration: Iteration number in search process (optional)
        parent_id: ID of parent architecture if this is a mutation (optional)
    """

    # Model architecture
    n_layer: int = 6
    n_embd: int = 384
    n_heads: int = 8

    # Attention configuration
    attention_type: str = "fhn"
    use_qk_norm: bool = True

    # Component choices
    use_mhc: bool = True
    use_mla: bool = False
    use_moe: bool = False
    use_kan: bool = True

    # KAN configuration
    kan_type: str = "faster"
    kan_num_centers: int = 3

    # FHN dynamics
    fhn_threshold: float = 0.5
    fhn_tau: float = 12.5
    use_fhn_parallel: bool = True

    # Manifold projection
    manifold_dim: int = 64
    n_eigenvectors: int = 32
    use_multiscale_manifold: bool = True

    # Regularization
    dropout: float = 0.0

    # Memory and SDR configuration
    use_sdr: bool = False
    sdr_size: int = 2048
    sdr_sparsity: float = 0.02
    engram_capacity: int = 1000
    engram_threshold: float = 0.3

    # Metadata (optional)
    architecture_id: Optional[str] = None
    search_iteration: Optional[int] = None
    parent_id: Optional[str] = None

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate the architecture configuration.

        Checks for common configuration errors such as:
        - n_embd must be divisible by n_heads
        - Valid choices for categorical parameters
        - Reasonable ranges for numerical parameters

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if configuration is valid, False otherwise
            - error_message: Description of validation error, None if valid
        """
        # Check n_embd divisibility by n_heads
        if self.n_embd % self.n_heads != 0:
            return (
                False,
                f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})",
            )

        # Validate attention_type
        valid_attention_types = ["fhn", "kaufmann", "knot"]
        if self.attention_type not in valid_attention_types:
            return (
                False,
                f"attention_type must be one of {valid_attention_types}, got {self.attention_type}",
            )

        # Validate kan_type
        valid_kan_types = ["faster", "wave", "cheby"]
        if self.kan_type not in valid_kan_types:
            return (
                False,
                f"kan_type must be one of {valid_kan_types}, got {self.kan_type}",
            )

        # Validate numerical ranges
        if self.n_layer < 1:
            return False, f"n_layer must be >= 1, got {self.n_layer}"

        if self.n_embd < 1:
            return False, f"n_embd must be >= 1, got {self.n_embd}"

        if self.n_heads < 1:
            return False, f"n_heads must be >= 1, got {self.n_heads}"

        if not (0.0 <= self.dropout <= 1.0):
            return False, f"dropout must be in [0, 1], got {self.dropout}"

        if self.fhn_threshold <= 0:
            return False, f"fhn_threshold must be > 0, got {self.fhn_threshold}"

        if self.fhn_tau <= 0:
            return False, f"fhn_tau must be > 0, got {self.fhn_tau}"

        if self.manifold_dim < 1:
            return False, f"manifold_dim must be >= 1, got {self.manifold_dim}"

        if self.n_eigenvectors < 1:
            return False, f"n_eigenvectors must be >= 1, got {self.n_eigenvectors}"

        if self.kan_num_centers < 1:
            return False, f"kan_num_centers must be >= 1, got {self.kan_num_centers}"

        # Validate SDR parameters
        if self.sdr_size < 1:
            return False, f"sdr_size must be >= 1, got {self.sdr_size}"

        if not (0.0 < self.sdr_sparsity <= 1.0):
            return False, f"sdr_sparsity must be in (0, 1], got {self.sdr_sparsity}"

        if self.engram_capacity < 1:
            return False, f"engram_capacity must be >= 1, got {self.engram_capacity}"

        if not (0.0 <= self.engram_threshold <= 1.0):
            return (
                False,
                f"engram_threshold must be in [0, 1], got {self.engram_threshold}",
            )

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "n_heads": self.n_heads,
            "attention_type": self.attention_type,
            "use_qk_norm": self.use_qk_norm,
            "use_mhc": self.use_mhc,
            "use_mla": self.use_mla,
            "use_moe": self.use_moe,
            "use_kan": self.use_kan,
            "kan_type": self.kan_type,
            "kan_num_centers": self.kan_num_centers,
            "fhn_threshold": self.fhn_threshold,
            "fhn_tau": self.fhn_tau,
            "use_fhn_parallel": self.use_fhn_parallel,
            "manifold_dim": self.manifold_dim,
            "n_eigenvectors": self.n_eigenvectors,
            "use_multiscale_manifold": self.use_multiscale_manifold,
            "dropout": self.dropout,
            # Memory/SDR configuration
            "use_sdr": self.use_sdr,
            "sdr_size": self.sdr_size,
            "sdr_sparsity": self.sdr_sparsity,
            "engram_capacity": self.engram_capacity,
            "engram_threshold": self.engram_threshold,
            # Metadata
            "architecture_id": self.architecture_id,
            "search_iteration": self.search_iteration,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ArchitectureConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            ArchitectureConfig instance
        """
        return cls(**config_dict)

    def to_config(
        self, vocab_size: int, block_size: int = 1024, **kwargs: Any
    ) -> "NeuroManifoldConfig":
        """Convert ArchitectureConfig to NeuroManifoldConfig for training.

        This method maps the architecture search parameters to a complete
        NeuroManifoldConfig that can be used to instantiate and train a model.

        Args:
            vocab_size: Vocabulary size (required)
            block_size: Maximum sequence length (default: 1024)
            **kwargs: Additional parameters to override defaults

        Returns:
            NeuroManifoldConfig instance ready for model instantiation

        Example:
            >>> arch = ArchitectureConfig(n_layer=6, n_embd=384)
            >>> config = arch.to_config(vocab_size=50304)
            >>> # Now config can be used to create a NeuroManifoldGPT model
        """
        from neuromanifold_gpt.config import NeuroManifoldConfig

        # Map ArchitectureConfig parameters to NeuroManifoldConfig
        config_dict = {
            # Vocabulary and sequence
            "vocab_size": vocab_size,
            "block_size": block_size,
            # Model architecture (from ArchitectureConfig)
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "n_heads": self.n_heads,
            # Attention configuration (from ArchitectureConfig)
            "attention_type": self.attention_type,
            "use_qk_norm": self.use_qk_norm,
            # Component choices (from ArchitectureConfig)
            "use_mhc": self.use_mhc,
            "use_mla": self.use_mla,
            "use_moe": self.use_moe,
            "use_kan": self.use_kan,
            # KAN configuration (from ArchitectureConfig)
            "kan_type": self.kan_type,
            "kan_num_centers": self.kan_num_centers,
            # FHN dynamics (from ArchitectureConfig)
            "fhn_threshold": self.fhn_threshold,
            "fhn_tau": self.fhn_tau,
            "use_fhn_parallel": self.use_fhn_parallel,
            # Manifold projection (from ArchitectureConfig)
            "manifold_dim": self.manifold_dim,
            "n_eigenvectors": self.n_eigenvectors,
            "use_multiscale_manifold": self.use_multiscale_manifold,
            # Regularization (from ArchitectureConfig)
            "dropout": self.dropout,
            # Memory/SDR configuration (from ArchitectureConfig)
            "use_sdr": self.use_sdr,
            "sdr_size": self.sdr_size,
            "sdr_sparsity": self.sdr_sparsity,
            "engram_capacity": self.engram_capacity,
            "engram_threshold": self.engram_threshold,
        }

        # Override with any additional kwargs
        config_dict.update(kwargs)

        return NeuroManifoldConfig(**config_dict)


class SearchSpace:
    """Neural architecture search space definition.

    This class defines the valid search space for NAS, including:
    - Discrete choices for categorical parameters
    - Continuous/discrete ranges for numerical parameters
    - Constraints and dependencies between parameters

    The search space can be used by various NAS algorithms including:
    - Random search
    - Evolutionary algorithms
    - Bayesian optimization
    - Reinforcement learning-based methods
    - Gradient-based NAS (DARTS, etc.)

    Example:
        >>> search_space = SearchSpace()
        >>> config = search_space.sample_random()
        >>> is_valid, error = config.validate()
        >>> if is_valid:
        ...     # Use config for training
        ...     pass
    """

    def __init__(self):
        """Initialize the search space with default ranges and choices."""
        # Model architecture ranges
        self.n_layer_choices = [4, 6, 8, 12]  # Discrete choices for depth
        self.n_embd_choices = [256, 384, 512, 768]  # Discrete choices for width
        self.n_heads_choices = [4, 6, 8, 12, 16]  # Must divide n_embd

        # Attention configuration
        self.attention_type_choices = ["fhn", "kaufmann", "knot"]
        self.use_qk_norm_choices = [True, False]

        # Component choices (boolean flags)
        self.use_mhc_choices = [True, False]
        self.use_mla_choices = [True, False]
        self.use_moe_choices = [True, False]
        self.use_kan_choices = [True, False]

        # KAN configuration
        self.kan_type_choices = ["faster", "wave", "cheby"]
        self.kan_num_centers_choices = [2, 3, 4, 5]

        # FHN dynamics
        self.fhn_threshold_range = (0.3, 0.7)  # Continuous range
        self.fhn_tau_choices = [10.0, 12.5, 15.0, 20.0]  # Discrete choices
        self.use_fhn_parallel_choices = [True, False]

        # Manifold projection
        self.manifold_dim_choices = [32, 64, 96, 128]
        self.n_eigenvectors_choices = [16, 32, 48, 64]
        self.use_multiscale_manifold_choices = [True, False]

        # Regularization
        self.dropout_range = (0.0, 0.2)  # Continuous range

        # Memory/SDR configuration
        self.use_sdr_choices = [True, False]
        self.sdr_size_choices = [1024, 2048, 4096]
        self.sdr_sparsity_choices = [0.01, 0.02, 0.03]  # 1-3% sparsity
        self.engram_capacity_choices = [500, 1000, 2000, 5000]
        self.engram_threshold_choices = [0.2, 0.3, 0.4]

    def sample(self) -> ArchitectureConfig:
        """Sample a random architecture from the search space.

        This is the main sampling method used by NAS algorithms.

        Returns:
            ArchitectureConfig with randomly sampled parameters

        Note:
            The sampled configuration is guaranteed to satisfy basic
            constraints (e.g., n_embd divisible by n_heads) and passes
            validation checks.
        """
        return self.sample_random()

    def sample_random(self) -> ArchitectureConfig:
        """Sample a random architecture from the search space.

        Returns:
            ArchitectureConfig with randomly sampled parameters

        Note:
            The sampled configuration is guaranteed to satisfy basic
            constraints (e.g., n_embd divisible by n_heads) but may
            need validation for more complex constraints.
        """
        import random

        # Sample basic architecture
        n_embd = random.choice(self.n_embd_choices)
        # Filter n_heads choices to only those that divide n_embd
        valid_n_heads = [h for h in self.n_heads_choices if n_embd % h == 0]
        n_heads = random.choice(valid_n_heads)
        n_layer = random.choice(self.n_layer_choices)

        # Sample attention configuration
        attention_type = random.choice(self.attention_type_choices)
        use_qk_norm = random.choice(self.use_qk_norm_choices)

        # Sample component choices
        use_mhc = random.choice(self.use_mhc_choices)
        use_mla = random.choice(self.use_mla_choices)
        use_moe = random.choice(self.use_moe_choices)
        use_kan = random.choice(self.use_kan_choices)

        # Sample KAN configuration
        kan_type = random.choice(self.kan_type_choices)
        kan_num_centers = random.choice(self.kan_num_centers_choices)

        # Sample FHN dynamics
        fhn_threshold = random.uniform(*self.fhn_threshold_range)
        fhn_tau = random.choice(self.fhn_tau_choices)
        use_fhn_parallel = random.choice(self.use_fhn_parallel_choices)

        # Sample manifold projection
        manifold_dim = random.choice(self.manifold_dim_choices)
        n_eigenvectors = random.choice(self.n_eigenvectors_choices)
        use_multiscale_manifold = random.choice(self.use_multiscale_manifold_choices)

        # Sample regularization
        dropout = random.uniform(*self.dropout_range)

        # Sample memory/SDR configuration
        use_sdr = random.choice(self.use_sdr_choices)
        sdr_size = random.choice(self.sdr_size_choices)
        sdr_sparsity = random.choice(self.sdr_sparsity_choices)
        engram_capacity = random.choice(self.engram_capacity_choices)
        engram_threshold = random.choice(self.engram_threshold_choices)

        return ArchitectureConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            n_heads=n_heads,
            attention_type=attention_type,
            use_qk_norm=use_qk_norm,
            use_mhc=use_mhc,
            use_mla=use_mla,
            use_moe=use_moe,
            use_kan=use_kan,
            kan_type=kan_type,
            kan_num_centers=kan_num_centers,
            fhn_threshold=fhn_threshold,
            fhn_tau=fhn_tau,
            use_fhn_parallel=use_fhn_parallel,
            manifold_dim=manifold_dim,
            n_eigenvectors=n_eigenvectors,
            use_multiscale_manifold=use_multiscale_manifold,
            dropout=dropout,
            use_sdr=use_sdr,
            sdr_size=sdr_size,
            sdr_sparsity=sdr_sparsity,
            engram_capacity=engram_capacity,
            engram_threshold=engram_threshold,
        )

    def get_default(self) -> ArchitectureConfig:
        """Get the default architecture configuration.

        Returns:
            ArchitectureConfig with default parameters
        """
        return ArchitectureConfig()

    def get_search_space_size(self) -> int:
        """Calculate the approximate size of the discrete search space.

        Returns:
            Approximate number of distinct architectures (ignoring continuous parameters)

        Note:
            The actual search space is much larger when considering
            continuous parameters like fhn_threshold and dropout.
        """
        # Calculate combinatorial size
        size = 1
        size *= len(self.n_layer_choices)
        size *= len(self.n_embd_choices)
        # n_heads depends on n_embd, approximate with average
        avg_valid_heads = sum(
            len([h for h in self.n_heads_choices if embd % h == 0])
            for embd in self.n_embd_choices
        ) / len(self.n_embd_choices)
        size *= int(avg_valid_heads)
        size *= len(self.attention_type_choices)
        size *= len(self.use_qk_norm_choices)
        size *= len(self.use_mhc_choices)
        size *= len(self.use_mla_choices)
        size *= len(self.use_moe_choices)
        size *= len(self.use_kan_choices)
        size *= len(self.kan_type_choices)
        size *= len(self.kan_num_centers_choices)
        size *= len(self.fhn_tau_choices)
        size *= len(self.use_fhn_parallel_choices)
        size *= len(self.manifold_dim_choices)
        size *= len(self.n_eigenvectors_choices)
        size *= len(self.use_multiscale_manifold_choices)
        size *= len(self.use_sdr_choices)
        size *= len(self.sdr_size_choices)
        size *= len(self.sdr_sparsity_choices)
        size *= len(self.engram_capacity_choices)
        size *= len(self.engram_threshold_choices)
        return int(size)

    def get_parameter_space(self) -> Dict[str, Union[List, Tuple]]:
        """Get dictionary representation of the search space.

        Returns:
            Dictionary mapping parameter names to their valid choices/ranges

        Example:
            >>> search_space = SearchSpace()
            >>> param_space = search_space.get_parameter_space()
            >>> print(param_space['n_layer'])
            [4, 6, 8, 12]
        """
        return {
            "n_layer": self.n_layer_choices,
            "n_embd": self.n_embd_choices,
            "n_heads": self.n_heads_choices,
            "attention_type": self.attention_type_choices,
            "use_qk_norm": self.use_qk_norm_choices,
            "use_mhc": self.use_mhc_choices,
            "use_mla": self.use_mla_choices,
            "use_moe": self.use_moe_choices,
            "use_kan": self.use_kan_choices,
            "kan_type": self.kan_type_choices,
            "kan_num_centers": self.kan_num_centers_choices,
            "fhn_threshold": self.fhn_threshold_range,
            "fhn_tau": self.fhn_tau_choices,
            "use_fhn_parallel": self.use_fhn_parallel_choices,
            "manifold_dim": self.manifold_dim_choices,
            "n_eigenvectors": self.n_eigenvectors_choices,
            "use_multiscale_manifold": self.use_multiscale_manifold_choices,
            "dropout": self.dropout_range,
            # Memory/SDR parameters
            "use_sdr": self.use_sdr_choices,
            "sdr_size": self.sdr_size_choices,
            "sdr_sparsity": self.sdr_sparsity_choices,
            "engram_capacity": self.engram_capacity_choices,
            "engram_threshold": self.engram_threshold_choices,
        }
