from __future__ import annotations

import torch


def compute_default_pooling(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute standard pooled embeddings from tokens in [T, N, D].
    Returns:
      pooled: [T, D] as mean over N
      global_pooled: [D] as mean over T
    """
    if tokens.ndim != 3:
        raise ValueError(f"Expected [T, N, D], got shape {tuple(tokens.shape)}")

    pooled = tokens.mean(dim=1)
    global_pooled = pooled.mean(dim=0)
    return pooled, global_pooled
