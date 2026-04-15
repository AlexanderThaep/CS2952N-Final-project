from __future__ import annotations

from pathlib import Path

import torch

from src.models.base import LatentBatchOutput


def save_latent_batch(output: LatentBatchOutput, output_root: str | Path) -> None:
    target_dir = Path(output_root) / output.model_name / output.sample_id
    target_dir.mkdir(parents=True, exist_ok=True)

    for layer in output.layers:
        payload = {
            "tokens": layer.tokens.cpu(),
            "pooled": layer.pooled.cpu(),
            "global_pooled": layer.global_pooled.cpu(),
            "metadata": layer.metadata,
        }
        torch.save(payload, target_dir / f"{layer.layer_name}.pt")
