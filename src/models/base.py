"""Shared interface for model-specific latent extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

@dataclass
class LayerLatentOutput:
    layer_name: str
    tokens: torch.Tensor  # [T, N, D]
    pooled: torch.Tensor  # [T, D]
    global_pooled: torch.Tensor  # [D]
    metadata: dict[str, Any]

@dataclass
class LatentBatchOutput:
    sample_id: str
    model_name: str
    layers: list[LayerLatentOutput]

class LatentExtractor(ABC):
    """Adapter interface for extracting standardized latent representations."""

    def __init__(self, model_name: str, checkpoint_path: str | None = None) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model = None

    @abstractmethod
    def load_pretrained(self) -> None:
        """Load model weights in frozen inference mode."""

    @abstractmethod
    def preprocess(self, sample: Any) -> Any:
        """Prepare a dataset sample for model forward pass."""

    @abstractmethod
    def forward_latents(self, batch: Any, layer_specs: list[str]) -> list[LayerLatentOutput]:
        """Run forward pass and return layer outputs in [T, N, D] format."""

    def postprocess(self, outputs: list[LayerLatentOutput]) -> list[LayerLatentOutput]:
        """Optional postprocessing hook."""
        return outputs

    def extract(self, sample_id: str, sample: Any, layer_specs: list[str]) -> LatentBatchOutput:
        if self.model is None:
            self.load_pretrained()

        batch = self.preprocess(sample)
        layer_outputs = self.forward_latents(batch, layer_specs)
        layer_outputs = self.postprocess(layer_outputs)
        return LatentBatchOutput(sample_id=sample_id, model_name=self.model_name, layers=layer_outputs)