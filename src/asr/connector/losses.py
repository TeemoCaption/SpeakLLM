"""Loss helpers used across duplex training phases."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    t_logits = teacher_logits / temperature
    s_logits = student_logits / temperature
    t_probs = F.softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    return F.kl_div(s_log_probs, t_probs, log_target=False, reduction="batchmean") * (temperature**2)


def ctc_loss(logits: torch.Tensor, targets: torch.Tensor, logit_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    return criterion(log_probs.transpose(0, 1), targets, logit_lengths, target_lengths)


def dtw_loss(embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
    """Dynamic time warping distance between embedding sequences."""
    cost = torch.cdist(embeddings, target_embeddings)
    T, U = cost.shape
    dp = torch.zeros((T + 1, U + 1), device=cost.device) + torch.inf
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(1, U + 1):
            dp[i, j] = cost[i - 1, j - 1] + torch.min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[T, U] / (T + U)


def roundtrip_consistency_loss(asr_logits: torch.Tensor, tts_logits: torch.Tensor) -> torch.Tensor:
    """Encourage ASR and TTS logits to agree after roundtrip."""
    asr_probs = F.softmax(asr_logits, dim=-1)
    tts_probs = F.softmax(tts_logits, dim=-1)
    return F.mse_loss(asr_probs, tts_probs)
