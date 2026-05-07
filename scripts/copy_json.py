#!/usr/bin/env python3

from pathlib import Path
import shutil

LATENTS_ROOT = Path("latents")
PROCESSED_ROOT = Path("data")

for latent_file in LATENTS_ROOT.rglob("render_traj_color_latents.pt"):
    traj_dir = latent_file.parent

    # Relative path from latents root
    rel = traj_dir.relative_to(LATENTS_ROOT)

    # Corresponding camera.json path in processed_data
    src_camera = PROCESSED_ROOT / rel / "cameras.json"

    # Destination path inside latents tree
    dst_camera = traj_dir / "cameras.json"

    if src_camera.exists():
        shutil.copy2(src_camera, dst_camera)
        print(f"Copied: {src_camera} -> {dst_camera}")
    else:
        print(f"Missing: {src_camera}")