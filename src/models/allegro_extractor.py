"""Allegro latent extractor stub."""

from __future__ import annotations

from typing import Any

import torch

from src.models.base import LatentExtractor, LayerLatentOutput
from src.utils.pooling import compute_default_pooling

from src.allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D

class AllegroExtractor(LatentExtractor):
    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu") -> None:
        super().__init__(model_name="allegro", checkpoint_path=checkpoint_path)
        self.device = device

    def load_pretrained(self) -> None:
        self.model = AllegroAutoencoderKL3D.from_pretrained(self.checkpoint_path, torch_dtype=torch.float32).cuda()

    def preprocess(self, sample: Any) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "modality": sample.modality,
            "source_path": sample.source_path,
        }

    def forward_latents(self, batch: Any, layer_specs: list[str]) -> list[LayerLatentOutput]:
        outputs: list[LayerLatentOutput] = []

        for layer_name in layer_specs:
            pass

        return outputs
