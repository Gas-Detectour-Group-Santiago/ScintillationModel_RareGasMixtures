from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..core.registry import ProjectRegistry


class NormalizationPolicy(Protocol):
    normalization_id: str
    def apply(self, values: np.ndarray, *, fit_name: str | None = None) -> np.ndarray: ...


@dataclass(frozen=True)
class AbsoluteNormalization:
    normalization_id: str = "absolute"
    scale: float = 1000.0
    def apply(self, values: np.ndarray, *, fit_name: str | None = None) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale


@dataclass(frozen=True)
class FitNormalization:
    project_root: Path
    normalization_id: str
    reference_fit: str
    scale: float = 1000.0

    def _nnorm(self) -> float:
        path = self.project_root / "outputs" / "current" / "fits" / "products" / f"{self.reference_fit}_central.csv"
        frame = pd.read_csv(path)
        if "name" in frame.columns:
            rows = frame.loc[frame["name"].astype(str).str.lower().isin({"nnorm", "n_norm", "normalization"})]
            if not rows.empty:
                for col in ("value", "parameter"):
                    if col in rows.columns: return float(rows.iloc[0][col])
        for col in ("value", "parameter"):
            if col in frame.columns: return float(frame.iloc[0][col])
        raise ValueError(f"Cannot read Nnorm from {path}")

    def apply(self, values: np.ndarray, *, fit_name: str | None = None) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale / self._nnorm()


def policy_from_registry(project_root: str | Path, normalization_id: str) -> NormalizationPolicy:
    root = Path(project_root).resolve()
    spec = ProjectRegistry.load(root).normalization(normalization_id)
    if spec.mode == "as_fit" and not spec.reference_fit:
        return AbsoluteNormalization(normalization_id=normalization_id, scale=spec.output_scale)
    reference = spec.reference_fit
    if not reference:
        raise ValueError(f"Normalization {normalization_id!r} requires an explicit reference fit")
    return FitNormalization(root, normalization_id, reference, spec.output_scale)
