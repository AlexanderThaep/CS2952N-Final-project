from __future__ import annotations

from pathlib import Path

import torch

from src.models.base import LatentBatchOutput
from src.models.base import LayerLatentOutput

def save_latent(output: LayerLatentOutput, output_root: str | Path) -> None:
    target_dir = Path(output_root) / output.model_name / output.sample_id
    target_dir.mkdir(parents=True, exist_ok=True)

    torch.save(output.layer_latent, target_dir / f"{output.layer_name}.pt")

def save_latent_batch(output: LatentBatchOutput, output_root: str | Path) -> None:
    target_dir = Path(output_root) / output.model_name / output.sample_id
    target_dir.mkdir(parents=True, exist_ok=True)

    for layer in output.layers:
        torch.save(layer.layer_latent, target_dir / f"{layer.layer_name}.pt")
