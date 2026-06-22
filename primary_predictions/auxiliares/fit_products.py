from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FitProduct:
    name: str
    parameter_names: list[str]
    central: np.ndarray
    stat_toys: np.ndarray
    syst_toys: np.ndarray


class FitProductStore:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.parameters_dir = self.project_root / "data" / "Parameters"
        self.fit_results_dir = self.project_root / "data" / "FitResults"
        self.legacy_toys_dir = self.project_root / "data" / "sistematic_stadistic_data"

    def load(self, fit_name: str) -> FitProduct:
        names, central = self._load_central(fit_name)
        stat = self._load_toys(fit_name, "stat", expected_names=names)
        syst = self._load_toys(fit_name, "syst", expected_names=names)
        return FitProduct(
            name=fit_name,
            parameter_names=names,
            central=central,
            stat_toys=stat,
            syst_toys=syst,
        )

    def _load_central(self, fit_name: str) -> tuple[list[str], np.ndarray]:
        candidates = [
            self.parameters_dir / f"{fit_name}.csv",
            self.fit_results_dir / f"{fit_name}_central.csv",
        ]

        for path in candidates:
            if not path.exists():
                continue

            df = pd.read_csv(path)

            if {"name", "value"}.issubset(df.columns):
                return df["name"].astype(str).tolist(), df["value"].to_numpy(dtype=float)

            if {"parameter_name", "value"}.issubset(df.columns):
                return df["parameter_name"].astype(str).tolist(), df["value"].to_numpy(dtype=float)

            # Legacy format: index = parameter names, column = parameter.
            df_idx = pd.read_csv(path, index_col=0)
            if "parameter" in df_idx.columns:
                return df_idx.index.astype(str).tolist(), df_idx["parameter"].to_numpy(dtype=float)

            if "value" in df_idx.columns:
                return df_idx.index.astype(str).tolist(), df_idx["value"].to_numpy(dtype=float)

        raise FileNotFoundError(
            f"No encuentro parámetros centrales para {fit_name!r}. "
            f"Ejecuta antes primary_fits/{fit_name} o revisa data/Parameters y data/FitResults."
        )

    def _load_toys(self, fit_name: str, kind: str, expected_names: list[str]) -> np.ndarray:
        path = self.fit_results_dir / f"{fit_name}_toys_{kind}.csv"
        if path.exists():
            df = pd.read_csv(path)
            cols = [c for c in expected_names if c in df.columns]
            if len(cols) == len(expected_names):
                return df[cols].to_numpy(dtype=float)
            numeric = df.drop(columns=[c for c in ("toy_id", "seed", "success") if c in df.columns], errors="ignore")
            return numeric.select_dtypes(include=[np.number]).to_numpy(dtype=float)

        # Legacy npz fallback.
        npz_candidates = [
            self.legacy_toys_dir / f"{fit_name}_toy_parameters.npz",
            self.legacy_toys_dir / f"{fit_name}_toys.npz",
            self.legacy_toys_dir / f"{fit_name}_{kind}_toy_parameters.npz",
        ]
        aliases = {
            "stat": ("stat", "stat_params", "statistical", "statistical_params"),
            "syst": ("syst", "syst_params", "systematic", "sistematic", "systematic_params"),
        }[kind]

        for candidate in npz_candidates:
            if not candidate.exists():
                continue
            payload = np.load(candidate, allow_pickle=True)
            for key in aliases:
                if key in payload.files:
                    arr = np.asarray(payload[key], dtype=float)
                    if arr.ndim == 2:
                        return arr

        return np.empty((0, len(expected_names)), dtype=float)

