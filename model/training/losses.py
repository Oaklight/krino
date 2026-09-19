"""Differentiable loss functions for calibration-aware training.

Three calibration objectives (issue #23 Phase A):
  - Brier score: proper scoring rule, mean squared error on probabilities
  - MMCE: Maximum Mean Calibration Error (Kumar et al. 2018)
  - Focal loss: down-weights easy examples, improves calibration indirectly

All functions accept pre-softmax logits and integer targets to match
the interface in supervised.py's compute_loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def brier_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Brier score as a differentiable loss.

    For multi-class: mean squared error between predicted probabilities
    and one-hot target vector, averaged over classes.

    Args:
        logits: [batch, n_classes] pre-softmax scores.
        target: [batch] integer class indices.

    Returns:
        Scalar loss.
    """
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(target, num_classes=logits.shape[-1]).float()
    return ((probs - one_hot) ** 2).sum(dim=-1).mean()


def brier_loss_binary(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Brier score for binary (noul) predictions.

    Args:
        logit: [batch, 1] pre-sigmoid logit.
        target: [batch, 1] float target in {0, 1}.

    Returns:
        Scalar loss.
    """
    prob = torch.sigmoid(logit)
    return ((prob - target) ** 2).mean()


def mmce_kernel_loss(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    kernel_bandwidth: float = 0.25,
) -> torch.Tensor:
    """MMCE² from pre-extracted confidence and correctness tensors.

    This is the core MMCE kernel computation, factored out so it can be
    called on batches of per-item confidence/correctness pairs collected
    across multiple forward passes (virtual batching).

    Args:
        confidences: [N] predicted confidence in own prediction (has grad).
        correctness: [N] whether prediction was correct (0 or 1, detached).
        kernel_bandwidth: bandwidth for the Laplacian kernel.

    Returns:
        Scalar MMCE² loss.
    """
    n = confidences.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=confidences.device, requires_grad=True)

    cal_error = confidences - correctness
    conf_diff = (confidences.unsqueeze(1) - confidences.unsqueeze(0)).abs()
    kernel = torch.exp(-conf_diff / kernel_bandwidth)
    mmce_sq = (cal_error.unsqueeze(1) * cal_error.unsqueeze(0) * kernel).mean()

    return mmce_sq


def mmce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    kernel_bandwidth: float = 0.25,
) -> torch.Tensor:
    """Maximum Mean Calibration Error (Kumar et al. 2018).

    Kernel-based calibration error that is differentiable and does not
    require binning. Uses a Laplacian kernel on confidence values.

    Args:
        logits: [batch, n_classes] pre-softmax scores.
        target: [batch] integer class indices.
        kernel_bandwidth: bandwidth for the Laplacian kernel.

    Returns:
        Scalar MMCE² loss (squared MMCE for gradient stability).
    """
    probs = F.softmax(logits, dim=-1)
    confidences = probs.max(dim=-1).values
    predictions = probs.argmax(dim=-1)
    correctness = (predictions == target).float()

    return mmce_kernel_loss(confidences, correctness, kernel_bandwidth)


def mmce_loss_binary(
    logit: torch.Tensor,
    target: torch.Tensor,
    kernel_bandwidth: float = 0.25,
) -> torch.Tensor:
    """MMCE for binary predictions.

    Args:
        logit: [batch, 1] pre-sigmoid logit.
        target: [batch, 1] float target in {0, 1}.
        kernel_bandwidth: bandwidth for the Laplacian kernel.

    Returns:
        Scalar MMCE² loss.
    """
    prob = torch.sigmoid(logit).squeeze(-1)
    predicted = (prob > 0.5).float()
    target_sq = target.squeeze(-1)
    confidence = torch.where(prob > 0.5, prob, 1.0 - prob)
    correctness = (predicted == target_sq).float()

    return mmce_kernel_loss(confidence, correctness, kernel_bandwidth)


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017) for multi-class classification.

    Scales the CE loss by (1 - p_t)^gamma, down-weighting easy examples
    where the model is already confident. Known to improve calibration.

    Args:
        logits: [batch, n_classes] pre-softmax scores.
        target: [batch] integer class indices.
        gamma: focusing parameter. gamma=0 recovers standard CE.
        label_smoothing: optional label smoothing factor.

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    if label_smoothing > 0:
        n_classes = logits.shape[-1]
        smooth_target = torch.full_like(probs, label_smoothing / max(n_classes - 1, 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - label_smoothing)
        # focal weight based on predicted probability of true class
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** gamma
        ce = -(smooth_target * log_probs).sum(dim=-1)
        return (focal_weight * ce).mean()

    # gather log-prob and prob of the true class
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t) ** gamma

    return (focal_weight * nll).mean()


def focal_loss_binary(
    logit: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for binary (noul) predictions.

    No label_smoothing parameter — smoothing a binary target toward 0.5
    fights the sigmoid, unlike multi-class where it redistributes mass.

    Args:
        logit: [batch, 1] pre-sigmoid logit.
        target: [batch, 1] float target in {0, 1}.
        gamma: focusing parameter.

    Returns:
        Scalar loss.
    """
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    prob = torch.sigmoid(logit)
    p_t = target * prob + (1.0 - target) * (1.0 - prob)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()
