"""Shared interface for model-specific latent extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

@dataclass
class LayerLatentOutput:
    layer_name: str
    layer_latent: torch.Tensor

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
        self.local_batch_size = 2
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

    @abstractmethod
    def final_latents(self, batch: Any) -> torch.Tensor:
        """Run full forward pass and return scaled posterior tensor"""

    def postprocess(self, outputs: list[LayerLatentOutput]) -> list[LayerLatentOutput]:
        """Optional postprocessing hook."""
        return outputs

    def extract_layers(self, sample_id: str, sample: Any, layer_specs: list[str]) -> LatentBatchOutput:
        batch = self.preprocess(sample)
        layer_outputs = self.forward_latents(batch, layer_specs)
        layer_outputs = self.postprocess(layer_outputs)

        return LatentBatchOutput(sample_id=sample_id, model_name=self.model_name, layers=layer_outputs)

    def extract(self, sample_id: str, sample: Any) -> LayerLatentOutput:
        batch = self.preprocess(sample)
        output = self.final_latents(batch)
        output = self.postprocess(output)

        return LayerLatentOutput(layer_name="posterior", layer_latent=output)