"""Allegro latent extractor stub."""

from __future__ import annotations

from typing import Any

import torch

from src.models.base import LatentExtractor, LayerLatentOutput
from src.utils.pooling import compute_default_pooling

from src.allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D

class AllegroExtractor(LatentExtractor):
    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu", local_batch_size: int = 2) -> None:
        super().__init__(model_name="allegro", checkpoint_path=checkpoint_path)
        self.device = device
        self.local_batch_size = local_batch_size
        self.load_pretrained()

        print(f"Device set to: {device}")

    def load_pretrained(self) -> None:
        self.model = AllegroAutoencoderKL3D.from_pretrained(self.checkpoint_path, torch_dtype=torch.float32).to(self.device)

    def preprocess(self, sample: Any) -> Any:
        return sample

    def final_latents(self, batch) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.model.encode(
                batch,
                local_batch_size=self.local_batch_size
            )

            latents = posterior.latent_dist.mean * self.model.scale_factor

        return latents

    def forward_latents(self, batch: Any, layer_specs: list[str]) -> list[LayerLatentOutput]:
        activations = {}
        handles = []
        outputs: list[LayerLatentOutput] = []

        def make_hook(name):
            print(f"Hooked to layer: {name}")
            def hook(module, inp, out):
                activations[name] = out.detach().cpu()
            return hook

        layer_map = {
            name: module
            for name, module in self.model.encoder.named_modules()
            if name != ""
        }

        for layer_name in layer_specs:
            if layer_name not in layer_map:
                raise ValueError(
                    f"Unknown layer: {layer_name}\n"
                    f"Available layers: {list(layer_map.keys())}"
                )

            handle = layer_map[layer_name].register_forward_hook(
                make_hook(layer_name)
            )
            handles.append(handle)

        try:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch

            with torch.no_grad():
                self.model.encoder(x)

            for layer_name in layer_specs:
                outputs.append(
                    LayerLatentOutput(
                        layer_name=layer_name,
                        layer_latent=activations[layer_name]
                    )
                )

        finally:
            for handle in handles:
                handle.remove()

        return outputs
