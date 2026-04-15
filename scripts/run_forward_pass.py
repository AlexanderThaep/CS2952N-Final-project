#!/usr/bin/env python3
"""Forward-pass latent extraction runner for dataset_v0."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml
from tqdm import tqdm

# Ensure `src` is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import DatasetV0
from src.models import build_extractor
from src.utils import save_latent_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent extraction forward pass.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset_v0 directory.")
    parser.add_argument("--output_root", type=str, required=True, help="Path for saved latents.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    dataset = DatasetV0(
        root=args.dataset_root,
        metadata_file=dataset_cfg["metadata_file"],
        split=dataset_cfg.get("split"),
    )

    extractor = build_extractor(
        model_name=model_cfg["name"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        device=runtime_cfg.get("device", "cpu"),
    )
    layer_specs = model_cfg.get("layer_specs", [])

    for sample in tqdm(dataset, desc="Extracting latents"):
        output = extractor.extract(sample_id=sample.sample_id, sample=sample, layer_specs=layer_specs)
        save_latent_batch(output, args.output_root)

    print(f"Saved latent outputs to: {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
