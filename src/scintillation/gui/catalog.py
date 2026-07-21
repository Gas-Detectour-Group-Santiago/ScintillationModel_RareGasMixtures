from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    label: str
    path: Path
    stage: str
    family: str
    description: str

    def load(self, *, nrows: int | None = None) -> pd.DataFrame:
        return pd.read_csv(self.path, nrows=nrows)

    def relative_path(self, root: str | Path) -> str:
        root_path = Path(root).resolve()
        try:
            return self.path.resolve().relative_to(root_path).as_posix()
        except ValueError:
            return self.path.resolve().as_posix()


def _human_name(path: Path) -> str:
    name = path.name
    for suffix in (".csv.gz", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ")


def _iter_tables(directory: Path, *, recursive: bool = False) -> list[Path]:
    patterns = ("*.csv", "*.csv.gz")
    paths: set[Path] = set()
    for pattern in patterns:
        iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
        paths.update(iterator)
    return sorted(paths)


def discover_primary_datasets(root: str | Path) -> tuple[DatasetInfo, ...]:
    root = Path(root).resolve()
    items: list[DatasetInfo] = []
    processed = root / "data" / "processed" / "primary"
    for path in _iter_tables(processed):
        family = "degrad_energy" if "energy_cases" in path.name else "degrad_populations"
        items.append(DatasetInfo(
            dataset_id=f"primary.processed.{path.name.replace('.', '_')}",
            label=f"Degrad · {_human_name(path)}",
            path=path,
            stage="primary",
            family=family,
            description="Processed Degrad populations or energy-dependent primary quantities.",
        ))
    bands = root / "data" / "cache" / "predictions" / "Bands"
    for path in _iter_tables(bands):
        items.append(DatasetInfo(
            dataset_id=f"primary.bands.{path.name.replace('.', '_')}",
            label=f"Prediction · {_human_name(path)}",
            path=path,
            stage="primary",
            family="prediction_bands",
            description="Primary prediction with cached statistical and systematic bands.",
        ))
    summaries = root / "data" / "cache" / "predictions"
    for path in _iter_tables(summaries):
        items.append(DatasetInfo(
            dataset_id=f"primary.summary.{path.name.replace('.', '_')}",
            label=f"Summary · {_human_name(path)}",
            path=path,
            stage="primary",
            family="summary",
            description="Compact primary prediction or comparison table.",
        ))
    return tuple(items)


def discover_secondary_datasets(root: str | Path) -> tuple[DatasetInfo, ...]:
    root = Path(root).resolve()
    items: list[DatasetInfo] = []
    catalog = root / "data" / "cache" / "secondary" / "simulation_catalog.csv.gz"
    if catalog.exists():
        items.append(DatasetInfo(
            dataset_id="secondary.simulation_catalog",
            label="Avalanche and hLevels simulation catalogue",
            path=catalog,
            stage="secondary",
            family="simulation_catalog",
            description="One row per ROOT with gain, E/p, Townsend quantities and mapped hLevels populations.",
        ))
    bands = root / "data" / "cache" / "predictions" / "Secondary" / "Bands"
    for path in _iter_tables(bands):
        items.append(DatasetInfo(
            dataset_id=f"secondary.bands.{path.name.replace('.', '_')}",
            label=f"Scintillation · {_human_name(path)}",
            path=path,
            stage="secondary",
            family="prediction_bands",
            description="Secondary scintillation prediction with cached bands and selection metadata.",
        ))
    metadata = root / "data" / "cache" / "predictions" / "Secondary" / "Metadata"
    for path in _iter_tables(metadata):
        items.append(DatasetInfo(
            dataset_id=f"secondary.metadata.{path.name.replace('.', '_')}",
            label=f"Metadata · {_human_name(path)}",
            path=path,
            stage="secondary",
            family="metadata",
            description="Compact gain and field metadata behind secondary comparisons.",
        ))
    return tuple(items)


def load_datasets(infos: Sequence[DatasetInfo]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for info in infos:
        frame = info.load().copy()
        frame["dataset_id"] = info.dataset_id
        frame["dataset_label"] = info.label
        frame["dataset_family"] = info.family
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.select_dtypes(include="number").columns if frame[column].notna().any()]


def categorical_columns(frame: pd.DataFrame, *, max_unique: int = 40) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        unique = values.nunique(dropna=True)
        if frame[column].dtype == object or unique <= max_unique:
            columns.append(str(column))
    return columns


def available_values(frame: pd.DataFrame, column: str) -> list[object]:
    if column not in frame:
        return []
    values = frame[column].dropna().unique().tolist()
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=lambda value: str(value))


def condition_table(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return pd.DataFrame()
    return frame[selected].drop_duplicates().sort_values(selected, kind="stable").reset_index(drop=True)
