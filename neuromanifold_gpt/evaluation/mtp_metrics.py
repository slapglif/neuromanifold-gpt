"""MTP (Multi-Token Prediction) accuracy evaluation metrics.

This module provides metrics for evaluating Multi-Token Prediction quality:
- Per-depth accuracy: Token prediction accuracy at each depth (t+1, t+2, t+3, ...)
- Top-k accuracy: Accuracy with k-best predictions considered
- Confidence statistics: Prediction confidence analysis

Multi-Token Prediction (MTP) is a training technique where the model predicts
multiple future tokens simultaneously (DeepSeek-V2, Meta's approach). This helps
with training efficiency and representation learning.

Reference: neuromanifold_gpt/model/gpt.py (use_mtp, mtp_n_predict)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MTPMetrics:
    """Compute evaluation metrics for Multi-Token Prediction.

    All methods work on logits tensors from MTP heads and target token sequences.
    """

    @staticmethod
    def compute_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        top_k: int = 1,
    ) -> Dict[str, float]:
        """Compute token prediction accuracy from logits.

        Args:
            logits: Prediction logits of shape (batch, seq_len, vocab_size)
            targets: Target token IDs of shape (batch, seq_len)
            top_k: Number of top predictions to consider (default: 1 for exact match)

        Returns:
            Dictionary with accuracy metrics:
                - accuracy: Top-k accuracy (fraction of correct predictions)
                - num_correct: Number of correct predictions
                - num_total: Total number of predictions
        """
        # Get top-k predictions
        # logits: (batch, seq_len, vocab_size)
        # top_k_preds: (batch, seq_len, k)
        _, top_k_preds = logits.topk(k=top_k, dim=-1)

        # Expand targets to compare with top-k predictions
        # targets: (batch, seq_len) -> (batch, seq_len, 1)
        targets_expanded = targets.unsqueeze(-1)

        # Check if target is in top-k predictions
        # correct: (batch, seq_len, k) -> (batch, seq_len)
        correct = (top_k_preds == targets_expanded).any(dim=-1)

        # Compute accuracy
        num_correct = correct.sum().item()
        num_total = correct.numel()
        accuracy = num_correct / num_total if num_total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'num_correct': num_correct,
            'num_total': num_total,
        }

    @staticmethod
    def compute_top_k_accuracies(
        logits: torch.Tensor,
        targets: torch.Tensor,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """Compute top-k accuracy for multiple k values.

        Args:
            logits: Prediction logits of shape (batch, seq_len, vocab_size)
            targets: Target token IDs of shape (batch, seq_len)
            k_values: List of k values to compute (default: [1, 5, 10])

        Returns:
            Dictionary with top-k accuracies:
                - top_1_accuracy: Exact match accuracy
                - top_5_accuracy: Top-5 accuracy
                - top_10_accuracy: Top-10 accuracy
                - (additional keys for other k values)
        """
        metrics = {}

        for k in k_values:
            # Ensure k doesn't exceed vocab size
            actual_k = min(k, logits.shape[-1])
            acc_metrics = MTPMetrics.compute_accuracy(logits, targets, top_k=actual_k)
            metrics[f'top_{k}_accuracy'] = acc_metrics['accuracy']

        return metrics

    @staticmethod
    def compute_confidence_statistics(logits: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of prediction confidence.

        Confidence is measured as the probability assigned to the most likely token.
        High confidence indicates the model is certain about its predictions.

        Args:
            logits: Prediction logits of shape (batch, seq_len, vocab_size)

        Returns:
            Dictionary with confidence statistics:
                - confidence_mean: Average confidence (max probability)
                - confidence_std: Standard deviation of confidence
                - confidence_min: Minimum confidence observed
                - confidence_max: Maximum confidence observed
                - entropy_mean: Mean prediction entropy (uncertainty)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Max probability (confidence in top prediction)
        max_probs = probs.max(dim=-1)[0]

        # Compute entropy: -sum(p * log(p))
        # Higher entropy = more uncertain predictions
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return {
            'confidence_mean': max_probs.mean().item(),
            'confidence_std': max_probs.std().item(),
            'confidence_min': max_probs.min().item(),
            'confidence_max': max_probs.max().item(),
            'entropy_mean': entropy.mean().item(),
        }

    @staticmethod
    def compute_per_depth_accuracy(
        mtp_logits_list: List[torch.Tensor],
        targets: torch.Tensor,
        depths: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Compute accuracy at each MTP prediction depth.

        In MTP, we predict tokens at multiple future positions:
        - Depth 1: t+1 (main prediction head)
        - Depth 2: t+2 (first auxiliary head)
        - Depth 3: t+3 (second auxiliary head)
        - etc.

        Args:
            mtp_logits_list: List of logits tensors, one per depth
                Each tensor has shape (batch, seq_len, vocab_size)
            targets: Target token IDs of shape (batch, seq_len + n_depths)
                Must be long enough to contain targets for all depths
            depths: Optional list of depth labels (default: [1, 2, 3, ...])

        Returns:
            Dictionary with per-depth accuracies:
                - depth_1_accuracy: Accuracy at depth 1 (t+1)
                - depth_2_accuracy: Accuracy at depth 2 (t+2)
                - depth_N_accuracy: Accuracy at depth N
                - mean_depth_accuracy: Average across all depths
        """
        if depths is None:
            depths = list(range(1, len(mtp_logits_list) + 1))

        metrics = {}
        accuracies = []

        for depth_idx, (logits, depth) in enumerate(zip(mtp_logits_list, depths)):
            # For depth d, we predict tokens at position t+d
            # So targets should be shifted by d positions
            # logits: (batch, seq_len, vocab_size) predicts targets[:, d:seq_len+d]

            seq_len = logits.shape[1]
            depth_targets = targets[:, depth_idx:depth_idx + seq_len]

            # Ensure targets match logits sequence length
            if depth_targets.shape[1] != seq_len:
                # Truncate or pad as needed
                min_len = min(depth_targets.shape[1], seq_len)
                logits = logits[:, :min_len, :]
                depth_targets = depth_targets[:, :min_len]

            # Compute accuracy for this depth
            acc_metrics = MTPMetrics.compute_accuracy(logits, depth_targets)
            metrics[f'depth_{depth}_accuracy'] = acc_metrics['accuracy']
            accuracies.append(acc_metrics['accuracy'])

        # Compute mean accuracy across all depths
        if accuracies:
            metrics['mean_depth_accuracy'] = sum(accuracies) / len(accuracies)

        return metrics

    @staticmethod
    def compute_all(
        logits: torch.Tensor,
        targets: torch.Tensor,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """Compute all MTP accuracy metrics.

        Args:
            logits: Prediction logits of shape (batch, seq_len, vocab_size)
            targets: Target token IDs of shape (batch, seq_len)
            k_values: List of k values for top-k accuracy (default: [1, 5, 10])

        Returns:
            Dictionary with all metrics:
                - accuracy: Top-1 accuracy
                - top_1_accuracy, top_5_accuracy, top_10_accuracy: Top-k accuracies
                - confidence_mean: Average prediction confidence
                - confidence_std: Standard deviation of confidence
                - confidence_min: Minimum confidence
                - confidence_max: Maximum confidence
                - entropy_mean: Mean prediction entropy
        """
        metrics = {}

        # Basic accuracy
        acc_metrics = MTPMetrics.compute_accuracy(logits, targets, top_k=1)
        metrics['accuracy'] = acc_metrics['accuracy']

        # Top-k accuracies
        topk_metrics = MTPMetrics.compute_top_k_accuracies(logits, targets, k_values)
        metrics.update(topk_metrics)

        # Confidence statistics
        conf_metrics = MTPMetrics.compute_confidence_statistics(logits)
        metrics.update(conf_metrics)

        return metrics
