from __future__ import annotations

from dataclasses import dataclass

@dataclass
class LinearProbeConfig:
    target_name: str
    input_dim: int
    output_dim: int
    normalize_input: bool = True