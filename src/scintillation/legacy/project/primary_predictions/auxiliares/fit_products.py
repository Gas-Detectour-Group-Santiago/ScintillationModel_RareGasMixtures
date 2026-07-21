from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scintillation.fitting.toy_cache import load_toys


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
        return load_toys(self.fit_results_dir, fit_name, kind, expected_names)
