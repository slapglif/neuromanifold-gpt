"""CPU-efficient SDR (Sparse Distributed Representations) operations.

SDRs are binary vectors with ~2% sparsity where semantic similarity = bit overlap.
These operations use bitwise semantics for 56x compression vs dense embeddings.

Key concepts:
- hard_topk: Enforce exact sparsity (k active bits)
- soft_topk: Differentiable version for training (straight-through estimator)
- overlap_count: Semantic similarity via bit intersection count
- union/intersection: Bitwise set operations
"""
import torch


class SDROperations:
    """Static methods for SDR manipulation.
    
    All operations work on tensors of any shape, operating on the last dimension.
    SDRs are represented as float tensors with values in {0.0, 1.0}.
    """

    @staticmethod
    def hard_topk(scores: torch.Tensor, n_active: int) -> torch.Tensor:
        """Select top-k scoring positions as active bits.
        
        Args:
            scores: Tensor of any shape, scores computed over last dimension
            n_active: Number of active bits to select (sparsity = n_active / dim)
            
        Returns:
            Binary tensor same shape as scores with exactly n_active ones per row
        """
        _, topk_idx = torch.topk(scores, n_active, dim=-1)
        sdr = torch.zeros_like(scores)
        sdr.scatter_(-1, topk_idx, 1.0)
        return sdr

    @staticmethod
    def soft_topk(
        scores: torch.Tensor,
        n_active: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Differentiable top-k using straight-through estimator.

        Forward pass: returns hard_topk result
        Backward pass: gradients flow through soft sigmoid approximation

        Args:
            scores: Tensor of any shape, scores computed over last dimension
            n_active: Number of active bits to select
            temperature: Controls sharpness of soft approximation

        Returns:
            Binary tensor with gradients that flow through
        """
        # Numerical stability: clamp temperature and normalize scores
        temp = max(temperature, 0.1)  # Prevent temperature from going too low

        # Normalize scores to prevent extreme sigmoid saturation
        scores_centered = scores - scores.mean(dim=-1, keepdim=True)
        scores_std = scores_centered.std(dim=-1, keepdim=True).clamp(min=1e-6)
        scores_norm = scores_centered / scores_std

        # Scale for sigmoid (keep in reasonable range to avoid gradient vanishing)
        scaled = scores_norm / temp
        scaled = scaled.clamp(-10.0, 10.0)  # Prevent extreme saturation

        soft = torch.sigmoid(scaled)
        hard = SDROperations.hard_topk(scores, n_active)
        # Straight-through estimator: forward uses hard, backward uses soft
        return hard + (soft - soft.detach())

    @staticmethod
    def overlap_count(sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> torch.Tensor:
        """Count overlapping active bits between two SDRs.
        
        This is the core semantic similarity primitive: more overlap = more similar.
        
        Args:
            sdr_a: Binary tensor
            sdr_b: Binary tensor (same shape as sdr_a)
            
        Returns:
            Count of overlapping bits (summed over last dimension)
        """
        return (sdr_a * sdr_b).sum(dim=-1)

    @staticmethod
    def union(sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> torch.Tensor:
        """Compute union of two SDRs (bitwise OR).
        
        Args:
            sdr_a: Binary tensor
            sdr_b: Binary tensor (same shape as sdr_a)
            
        Returns:
            Binary tensor with 1 where either input has 1
        """
        return torch.clamp(sdr_a + sdr_b, max=1.0)

    @staticmethod
    def intersection(sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> torch.Tensor:
        """Compute intersection of two SDRs (bitwise AND).
        
        Args:
            sdr_a: Binary tensor
            sdr_b: Binary tensor (same shape as sdr_a)
            
        Returns:
            Binary tensor with 1 where both inputs have 1
        """
        return sdr_a * sdr_b

    @staticmethod
    def semantic_similarity(
        sdr_a: torch.Tensor, 
        sdr_b: torch.Tensor, 
        n_active: int
    ) -> torch.Tensor:
        """Compute semantic similarity as overlap fraction.
        
        Similarity = overlap_count / n_active
        - 1.0 = identical SDRs
        - 0.0 = completely disjoint SDRs
        
        Args:
            sdr_a: Binary tensor
            sdr_b: Binary tensor (same shape as sdr_a)
            n_active: Number of active bits (for normalization)
            
        Returns:
            Similarity score in [0, 1]
        """
        return SDROperations.overlap_count(sdr_a, sdr_b) / n_active

    @staticmethod
    def sparsity(sdr: torch.Tensor) -> torch.Tensor:
        """Compute sparsity of an SDR (fraction of active bits).
        
        Target sparsity for semantic folding is ~2% (40/2048).
        
        Args:
            sdr: Binary tensor
            
        Returns:
            Sparsity ratio (active bits / total bits)
        """
        return sdr.sum(dim=-1) / sdr.shape[-1]
