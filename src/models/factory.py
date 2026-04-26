from __future__ import annotations

from src.models.allegro_extractor import AllegroExtractor
from src.models.base import LatentExtractor
from src.models.videocrafter_extractor import VideoCrafterExtractor

def build_extractor(model_name: str, checkpoint_path: str | None = None, device: str = "cpu") -> LatentExtractor:
    model_name = model_name.lower()
    if model_name == "videocrafter":
        return VideoCrafterExtractor(checkpoint_path=checkpoint_path, device=device)
    if model_name == "allegro":
        return AllegroExtractor(checkpoint_path=checkpoint_path, device=device)

    raise ValueError(f"Unknown model name: {model_name}")