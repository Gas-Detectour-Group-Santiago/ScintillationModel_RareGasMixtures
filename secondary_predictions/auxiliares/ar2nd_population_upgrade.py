from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ("Ar_1s4_1s5", "Ar_1s2_1s3", "Ar_dbleStar")
_BINS = {
    "Ar_1s4_1s5": (11.50, 11.70),
    "Ar_1s2_1s3": (11.70, 12.00),
    "Ar_dbleStar": (12.00, 100.00),
}


@dataclass(frozen=True)
class UpgradeReport:
    population_csv: Path
    updated_rows: int
    missing_level_csvs: tuple[str, ...] = ()


def _normalise_gas(value: object) -> str:
    text = str(value).strip().lower()
    return {"argon": "ar", "ar": "ar"}.get(text, text)


def _level_csv_for_root(level_dir: Path, root_filename: str) -> Path | None:
    stem = Path(str(root_filename)).stem
    pattern = re.compile(rf"^{re.escape(stem)}(?:_(\d+))?\.csv$")
    candidates: list[tuple[int, int, Path]] = []
    for path in level_dir.glob(f"{stem}*.csv"):
        match = pattern.match(path.name)
        if match is None:
            continue
        suffix = int(match.group(1) or 1)
        candidates.append((suffix, path.stat().st_mtime_ns, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _argon_excitation_count(level_df: pd.DataFrame, lo_ev: float, hi_ev: float) -> float:
    required = {"gas", "state_name", "energy_eV", "n_events"}
    missing = required - set(level_df.columns)
    if missing:
        raise ValueError(f"La tabla de niveles no contiene {sorted(missing)}")

    gas = level_df["gas"].map(_normalise_gas)
    state = level_df["state_name"].fillna("").astype(str)
    energy = pd.to_numeric(level_df["energy_eV"], errors="coerce")
    events = pd.to_numeric(level_df["n_events"], errors="coerce").fillna(0.0)
    mask = (
        gas.eq("ar")
        & state.str.contains("EXC", case=False, regex=False)
        & energy.ge(float(lo_ev))
        & energy.lt(float(hi_ev))
    )
    return float(events.loc[mask].sum())


def upgrade_population_csv(population_csv: str | Path, *, force: bool = False) -> UpgradeReport:
    """Add the three disjoint Ar2nd population bins using saved level CSVs.

    This intentionally does not need uproot.  It reuses the per-ROOT level
    tables already exported by ``Analysis_secondary_garfield.py`` and updates
    the corresponding population summary in place.
    """
    population_csv = Path(population_csv)
    if not population_csv.is_file():
        return UpgradeReport(population_csv, 0, ())

    df = pd.read_csv(population_csv)
    if not force and all(column in df.columns for column in _REQUIRED_COLUMNS):
        # Old summaries may contain Ar_dbleStar integrated from 12.9 eV.  A
        # small sidecar marker distinguishes upgraded 12.0-eV summaries.
        marker = population_csv.with_suffix(population_csv.suffix + ".ar2nd_v2")
        if marker.is_file():
            return UpgradeReport(population_csv, 0, ())

    level_dir = population_csv.parent.parent / "csv"
    if not level_dir.is_dir():
        return UpgradeReport(population_csv, 0, tuple(df.get("file", pd.Series(dtype=str)).astype(str)))

    updated_rows = 0
    missing_files: list[str] = []
    for index, row in df.iterrows():
        root_filename = str(row.get("file", ""))
        level_csv = _level_csv_for_root(level_dir, root_filename)
        if level_csv is None:
            missing_files.append(root_filename)
            continue
        level_df = pd.read_csv(level_csv)
        for column, (lo_ev, hi_ev) in _BINS.items():
            value = _argon_excitation_count(level_df, lo_ev, hi_ev)
            df.loc[index, column] = value
            df.loc[index, f"Err{column}"] = np.sqrt(max(value, 0.0))
        updated_rows += 1

    if updated_rows:
        df.to_csv(population_csv, index=False)
        base_cols = [
            col
            for col in (
                "concentration",
                "electric_field",
                "gap_mm",
                "pressure",
                "npe",
                "ne",
                "ni",
                "ne_std",
                "ni_std",
                "n_entries",
            )
            if col in df.columns
        ]
        for column in _REQUIRED_COLUMNS:
            cols = base_cols + [column]
            err_column = f"Err{column}"
            if err_column in df.columns:
                cols.append(err_column)
            df[cols].to_csv(population_csv.parent / f"{column}.csv", index=False)
        population_csv.with_suffix(population_csv.suffix + ".ar2nd_v2").write_text(
            "Ar2nd bins: [11.5,11.7), [11.7,12.0), [12.0,100.0) eV\n"
        )

    return UpgradeReport(population_csv, updated_rows, tuple(missing_files))


def ensure_secondary_ar2nd_populations(project_root: str | Path) -> list[UpgradeReport]:
    root = Path(project_root)
    candidates = [
        root / "data" / "Secondary_GarfieldData" / "ArCF4" / "populations" / "ArCF4_secondary.csv",
        *sorted(
            (root / "data" / "Secondary_GarfieldData" / "ArCF4_paper").glob(
                "*/populations/ArCF4_secondary.csv"
            )
        ),
    ]
    reports = [upgrade_population_csv(path) for path in candidates if path.is_file()]
    return reports
