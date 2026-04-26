"""VideoCrafter latent extractor stub."""

from __future__ import annotations

from typing import Any

import torch

from src.models.base import LatentExtractor, LayerLatentOutput
from src.utils.pooling import compute_default_pooling

class VideoCrafterExtractor(LatentExtractor):
    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu") -> None:
        super().__init__(model_name="videocrafter", checkpoint_path=checkpoint_path)
        self.device = device

    def load_pretrained(self) -> None:
        """
        TODO: Replace this stub with real VideoCrafter checkpoint loading.
        This method intentionally keeps interface behavior stable for now.
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

        # Placeholder shapes for interface testing.
        # Replace with real hook outputs from VideoCrafter components.
        T, N, D = 16, 256, 768
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
                        "model": "videocrafter",
                        "shape": [T, N, D],
                        "source_path": batch["source_path"],
                        "note": "stub output; replace with actual activations",
                    },
                )
            )
        return outputs