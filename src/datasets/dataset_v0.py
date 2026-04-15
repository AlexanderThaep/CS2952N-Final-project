"""Dataset loader for metadata-first dataset_v0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetSample:
    sample_id: str
    source_path: str
    modality: str
    camera_intrinsics: list[list[float]] | None
    camera_extrinsics: list[list[float]] | None
    split: str
    raw: dict[str, Any]


class DatasetV0:
    def __init__(self, root: str | Path, metadata_file: str, split: str | None = None) -> None:
        self.root = Path(root)
        self.metadata_path = self.root / metadata_file
        self.split = split
        self.samples = self._load_samples()

    def _load_samples(self) -> list[DatasetSample]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        samples: list[DatasetSample] = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                sample = DatasetSample(
                    sample_id=record["sample_id"],
                    source_path=record["source_path"],
                    modality=record["modality"],
                    camera_intrinsics=record.get("camera_intrinsics"),
                    camera_extrinsics=record.get("camera_extrinsics"),
                    split=record.get("split", "train"),
                    raw=record,
                )
                if self.split is None or sample.split == self.split:
                    samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]
