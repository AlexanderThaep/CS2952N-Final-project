from __future__ import annotations

from src.models.base import LatentExtractor
from src.models.allegro_extractor import AllegroExtractor

def build_extractor(
        model_name: str, 
        checkpoint_path: str | None = None, 
        device: str = "cpu",
        local_batch_size: int = 2) -> LatentExtractor:
    model_name = model_name.lower()
    if model_name == "allegro":
        return AllegroExtractor(checkpoint_path=checkpoint_path, device=device, local_batch_size=local_batch_size)

    raise ValueError(f"Unknown model name: {model_name}")