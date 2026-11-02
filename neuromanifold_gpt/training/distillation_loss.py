# neuromanifold_gpt/training/distillation_loss.py
"""
Knowledge distillation loss functions for model compression.

Provides:
- kl_divergence_loss: KL divergence between teacher and student logits
- distillation_loss: Combined hard label CE + soft target KL loss

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
"""
import torch
import torch.nn.functional as F


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute KL divergence between teacher and student distributions.

    Uses temperature scaling to soften probability distributions, making
    the teacher's "dark knowledge" (relative probabilities) more accessible.

    Args:
        student_logits: (B, T, V) student model output logits
        teacher_logits: (B, T, V) teacher model output logits
        temperature: Temperature for softening distributions (higher = softer)

    Returns:
        kl_loss: Scalar KL divergence loss

    Notes:
        - Higher temperature reveals more information about similarities between classes
        - Temperature = 1.0 is standard softmax
        - Typical values: 2.0 - 4.0 for distillation
    """
    # Soften distributions with temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence: sum over vocab, mean over batch and sequence
    # KL(P || Q) = sum(P * log(P / Q)) = sum(P * (log P - log Q))
    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean',
        log_target=False,
    )

    return kl_loss


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Combined distillation loss with hard labels and soft targets.

    Combines:
    1. Hard label cross-entropy (learn correct predictions)
    2. Soft target KL divergence (learn teacher's distribution)

    Loss = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(teacher || student)

    The T^2 scaling ensures gradients from soft and hard targets are roughly balanced,
    since soft target gradients scale as 1/T^2.

    Args:
        student_logits: (B, T, V) student model output logits
        teacher_logits: (B, T, V) teacher model output logits
        labels: (B, T) ground truth token indices
        alpha: Weight for hard label loss (0.0 = only soft targets, 1.0 = only hard labels)
        temperature: Temperature for soft targets (higher = more information transfer)
        label_smoothing: Label smoothing for hard label CE loss
        ignore_index: Index to ignore in hard label CE loss (e.g., padding)

    Returns:
        loss: Scalar combined distillation loss

    Notes:
        - Typical alpha: 0.1 - 0.5 (more weight on soft targets helps transfer)
        - Temperature 2.0 - 4.0 works well for most cases
        - Higher temperature transfers more "dark knowledge" about class similarities
    """
    B, T, V = student_logits.shape

    # Hard label loss: standard cross-entropy
    hard_loss = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    # Soft target loss: KL divergence with temperature scaling
    # Note: teacher_logits should be detached to prevent backprop through teacher
    soft_loss = kl_divergence_loss(
        student_logits,
        teacher_logits.detach(),
        temperature=temperature,
    )

    # Combined loss with temperature^2 scaling for soft targets
    # This balances gradient magnitudes from hard and soft losses
    loss = alpha * hard_loss + (1.0 - alpha) * (temperature ** 2) * soft_loss

    return loss
