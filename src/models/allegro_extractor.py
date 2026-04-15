"""Allegro latent extractor stub."""

from __future__ import annotations

from typing import Any

import torch

from src.models.base import LatentExtractor, LayerLatentOutput
from src.utils.pooling import compute_default_pooling


class AllegroExtractor(LatentExtractor):
    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu") -> None:
        super().__init__(model_name="allegro", checkpoint_path=checkpoint_path)
        self.device = device

    def load_pretrained(self) -> None:
        """
        TODO: Replace with actual Allegro + NequIP checkpoint loading.
        """
        self.model = {"loaded": True, "checkpoint": self.checkpoint_path}

    def preprocess(self, sample: Any) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "modality": sample.modality,
            "source_path": sample.source_path,
        }

    def forward_latents(self, batch: Any, layer_specs: list[str]) -> list[LayerLatentOutput]:
        outputs: list[LayerLatentOutput] = []

        #For non-temporal modalities, use T=1 for compatibility.
        T, N, D = 1, 128, 256
        for layer_name in layer_specs:
            tokens = torch.randn(T, N, D, device=self.device)
            pooled, global_pooled = compute_default_pooling(tokens)
            outputs.append(
                LayerLatentOutput(
                    layer_name=layer_name,
                    tokens=tokens,
                    pooled=pooled,
                    global_pooled=global_pooled,
                    metadata={
                        "model": "allegro",
                        "shape": [T, N, D],
                        "source_path": batch["source_path"],
                        "note": "stub output; replace with actual activations",
                    },
                )
            )
        return outputs
