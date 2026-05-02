"""Train/test split utilities for the scene trajectory dataset.

Data layout assumed:
    data_root/
        <scene>/
            traj_<nnnnn>/
                cameras.json   # contains "motion_type" key
                render_traj_color.mp4

Two splitting strategies are provided:
  1. split_by_percentage  – motion-aware: splits within each motion-type bucket
                            so every motion type is represented in both splits.
  2. split_sklearn        – uses sklearn's train_test_split with stratification
                            on (scene, motion_type) to achieve the same guarantee.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrajRecord:
    scene: str
    traj_id: str          # e.g. "traj_00001"
    motion_type: str
    video_path: Path      # absolute path to render_traj_color.mp4
    cameras_path: Path    # absolute path to cameras.json
    split: Literal["train", "test"] = "train"


Split = dict[Literal["train", "test"], list[TrajRecord]]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _load_cameras(cameras_path: Path) -> dict:
    with cameras_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def discover_trajectories(data_root: str | Path) -> list[TrajRecord]:
    """Return all TrajRecords found under *data_root*, sorted by (scene, traj_id)."""
    data_root = Path(data_root)
    records: list[TrajRecord] = []

    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("."):
            continue
        for traj_dir in sorted(scene_dir.iterdir()):
            if not traj_dir.is_dir() or not traj_dir.name.startswith("traj_"):
                continue
            cameras_path = traj_dir / "cameras.json"
            video_path = traj_dir / "render_traj_color.mp4"
            if not cameras_path.exists():
                continue
            meta = _load_cameras(cameras_path)
            records.append(TrajRecord(
                scene=scene_dir.name,
                traj_id=traj_dir.name,
                motion_type=meta["motion_type"],
                video_path=video_path,
                cameras_path=cameras_path,
            ))

    return records


# ---------------------------------------------------------------------------
# Split 1: motion-aware percentage split
# ---------------------------------------------------------------------------

def split_by_percentage(
    data_root: str | Path,
    test_size: float = 0.2,
    *,
    scenes: list[str] | None = None,
) -> Split:
    """Split trajectories by percentage, stratified by (scene, motion_type).

    Within each (scene, motion_type) bucket the trajectories are kept in
    sorted order and the *last* ``test_size`` fraction is assigned to test.
    This mirrors a temporal/sequential split: earlier trajectories train,
    later ones test.

    Args:
        data_root:  Root directory containing scene subdirectories.
        test_size:  Fraction of trajectories to assign to the test split
                    (0 < test_size < 1).
        scenes:     If given, only include these scene names.

    Returns:
        A dict with "train" and "test" keys, each holding a list of
        :class:`TrajRecord` instances.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    records = discover_trajectories(data_root)
    if scenes is not None:
        records = [r for r in records if r.scene in scenes]

    # Group by (scene, motion_type)
    buckets: dict[tuple[str, str], list[TrajRecord]] = {}
    for rec in records:
        key = (rec.scene, rec.motion_type)
        buckets.setdefault(key, []).append(rec)

    train_records: list[TrajRecord] = []
    test_records: list[TrajRecord] = []

    for (scene, motion_type), bucket in sorted(buckets.items()):
        n = len(bucket)
        n_test = max(1, math.floor(n * test_size))
        n_train = n - n_test

        for rec in bucket[:n_train]:
            rec.split = "train"
            train_records.append(rec)
        for rec in bucket[n_train:]:
            rec.split = "test"
            test_records.append(rec)

    return {"train": train_records, "test": test_records}


# ---------------------------------------------------------------------------
# Split 2: sklearn stratified split
# ---------------------------------------------------------------------------

def split_sklearn(
    data_root: str | Path,
    test_size: float = 0.2,
    *,
    scenes: list[str] | None = None,
    random_state: int = 42,
) -> Split:
    """Split trajectories using :func:`sklearn.model_selection.train_test_split`.

    Stratification is done on a ``(scene, motion_type)`` composite label so
    that every (scene, motion_type) combination is proportionally represented
    in both splits.

    Args:
        data_root:      Root directory containing scene subdirectories.
        test_size:      Fraction of trajectories to assign to the test split.
        scenes:         If given, only include these scene names.
        random_state:   Seed for reproducibility.

    Returns:
        A dict with "train" and "test" keys, each holding a list of
        :class:`TrajRecord` instances.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    records = discover_trajectories(data_root)
    if scenes is not None:
        records = [r for r in records if r.scene in scenes]

    if len(records) < 2:
        raise ValueError("Need at least 2 trajectories to split.")

    # Composite stratification label
    labels = [f"{r.scene}__{r.motion_type}" for r in records]

    train_recs, test_recs = train_test_split(
        records,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    for rec in train_recs:
        rec.split = "train"
    for rec in test_recs:
        rec.split = "test"

    return {"train": list(train_recs), "test": list(test_recs)}


# ---------------------------------------------------------------------------
# Convenience: print a summary of a split
# ---------------------------------------------------------------------------

def summarize_split(split: Split) -> None:
    """Print per-scene, per-motion-type counts for each split."""
    for split_name in ("train", "test"):
        records = split.get(split_name, [])
        print(f"\n[{split_name}]  {len(records)} trajectories")
        # count by (scene, motion_type)
        counts: dict[tuple[str, str], int] = {}
        for rec in records:
            key = (rec.scene, rec.motion_type)
            counts[key] = counts.get(key, 0) + 1
        for (scene, mt), cnt in sorted(counts.items()):
            print(f"  {scene:20s}  {mt:20s}  {cnt}")
