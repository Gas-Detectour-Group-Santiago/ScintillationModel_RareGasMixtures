from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import numpy as np
import pandas as pd


def _numeric_toys(path: Path, names: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    if all(name in frame.columns for name in names):
        return frame[names].to_numpy(dtype=float, copy=True)
    numeric = frame.drop(columns=[c for c in ("toy_id", "seed", "success") if c in frame], errors="ignore")
    return numeric.select_dtypes(include=[np.number]).to_numpy(dtype=float, copy=True)


def compact_fit_toys(fit_dir: str | Path, *, remove_csv: bool = True) -> list[Path]:
    fit_dir = Path(fit_dir)
    written: list[Path] = []
    for central_path in sorted(fit_dir.glob("*_central.csv")):
        fit_name = central_path.name.removesuffix("_central.csv")
        central = pd.read_csv(central_path)
        if "name" in central.columns:
            names = central["name"].astype(str).tolist()
        elif "parameter_name" in central.columns:
            names = central["parameter_name"].astype(str).tolist()
        else:
            names = pd.read_csv(central_path, index_col=0).index.astype(str).tolist()
        arrays = {}
        for kind in ("stat", "syst"):
            csv_path = fit_dir / f"{fit_name}_toys_{kind}.csv"
            arrays[kind] = _numeric_toys(csv_path, names) if csv_path.exists() else np.empty((0, len(names)))
        out = fit_dir / f"{fit_name}_toys.npz"
        np.savez_compressed(out, parameter_names=np.asarray(names, dtype=str), stat=arrays["stat"], syst=arrays["syst"])
        written.append(out)
        if remove_csv:
            for kind in ("stat", "syst"):
                (fit_dir / f"{fit_name}_toys_{kind}.csv").unlink(missing_ok=True)
    clear_toy_cache()
    return written


@lru_cache(maxsize=32)
def _load_npz_cached(path_text: str, mtime_ns: int) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    with np.load(path_text, allow_pickle=False) as payload:
        names = tuple(str(v) for v in payload["parameter_names"].tolist())
        stat = np.asarray(payload["stat"], dtype=float)
        syst = np.asarray(payload["syst"], dtype=float)
    return names, stat, syst


def load_toys(fit_dir: str | Path, fit_name: str, kind: str, expected_names: list[str]) -> np.ndarray:
    fit_dir = Path(fit_dir)
    npz = fit_dir / f"{fit_name}_toys.npz"
    if npz.exists():
        names, stat, syst = _load_npz_cached(str(npz), npz.stat().st_mtime_ns)
        arr = stat if kind == "stat" else syst
        if list(names) == list(expected_names): return arr.copy()
        index = {name: i for i, name in enumerate(names)}
        if all(name in index for name in expected_names):
            return arr[:, [index[name] for name in expected_names]].copy()
    csv_path = fit_dir / f"{fit_name}_toys_{kind}.csv"
    if csv_path.exists():
        return _numeric_toys(csv_path, expected_names)
    return np.empty((0, len(expected_names)), dtype=float)


def clear_toy_cache() -> None:
    _load_npz_cached.cache_clear()


def prune_fit_cache(fit_dir: str | Path) -> None:
    """Keep only files reused by predictions plus central covariance products."""
    fit_dir = Path(fit_dir)
    keep_suffixes = ("_central.csv", "_covariance.csv", "_correlation.csv", "_toys.npz")
    for path in fit_dir.iterdir():
        if not path.is_file():
            continue
        if any(path.name.endswith(suffix) for suffix in keep_suffixes):
            continue
        path.unlink()
