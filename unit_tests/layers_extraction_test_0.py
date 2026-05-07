#!/usr/bin/env python3
from __future__ import annotations

import yaml
import torch
from pathlib import Path
from decord import VideoReader
from einops import rearrange

from src.models import build_extractor
from src.utils import save_latent_batch

def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def main() -> None:
    cfg = load_config("configs/allegro.yaml")

    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    extractor = build_extractor(
        model_name=model_cfg["name"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        device=runtime_cfg.get("device", "cpu"),
        local_batch_size=runtime_cfg["batch_size"]
    )
    layer_specs = model_cfg.get("layer_specs", [])

    vr = VideoReader("resources/demo_video.mp4")

    frames = vr.get_batch(range(len(vr))).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0
    frames = frames * 2.0 - 1.0
    frames = rearrange(frames, 'f h w c -> 1 c f h w')

    cube = frames[:,:,:24,:320,:320]
    cube = cube.cuda().to(torch.float32)

    layers_output = extractor.extract_layers(sample_id="", sample=cube, layer_specs=layer_specs)
    for l in layers_output.layers:
        print(f"Layer name: {l.layer_name} with shape: {l.layer_latent.shape}")

if __name__ == "__main__":
    main()
