from __future__ import annotations

from hashlib import sha1, sha256
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ..core.outputs import OutputManager
from ..core.paths import ProjectPaths


def _geometry_from_path(path: Path) -> str:
    key = "/".join(part.lower() for part in path.parts)
    if "thgem" in key: return "THGEM"
    if re.search(r"(^|/)gem([_/]|$)", key) or "gem_" in key: return "GEM"
    if "electricfield" in key or "uniform" in key: return "UNIFORM"
    return "UNSPECIFIED"


def _campaign_from_path(base: Path, path: Path) -> str:
    relative = path.relative_to(base)
    parts = list(relative.parts[:-2]) if len(relative.parts) >= 2 else list(relative.parts[:-1])
    return "/".join(parts) or "default"


def _stable_id(relative_csv: str, source_row: int, file_name: str) -> str:
    return sha1(f"{relative_csv}|{source_row}|{file_name}".encode()).hexdigest()[:16]


def _root_index(paths: ProjectPaths) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in paths.raw.joinpath("garfield").rglob("*.root"):
        index.setdefault(path.name, []).append(path)
    return index


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("electric_field", "pressure", "gap_mm", "npe", "ne", "ne_std", "ni", "ni_std", "concentration"):
        if column not in out: out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["concentration_percent"] = out["concentration"]
    out["concentration_fraction"] = out["concentration_percent"] / 100.0
    out["electric_field_kV_cm"] = out["electric_field"]
    out["reduced_field_kV_cm_bar"] = out["electric_field_kV_cm"] / out["pressure"].replace(0.0, np.nan)
    out["gain"] = out["ne"]
    out["gain_std"] = out["ne_std"]
    out["gain_resolution"] = out["gain_std"] / out["gain"].replace(0.0, np.nan)
    out["gap_cm"] = out["gap_mm"] / 10.0
    positive_gain = out["gain"].where(out["gain"] > 0.0)
    out["alpha_eff_cm_inv"] = np.log(positive_gain) / out["gap_cm"].replace(0.0, np.nan)
    out["alpha_eff_over_p_cm_inv_bar"] = out["alpha_eff_cm_inv"] / out["pressure"].replace(0.0, np.nan)
    out["ion_electron_ratio"] = out["ni"] / out["ne"].replace(0.0, np.nan)
    out["charge_imbalance_fraction"] = (out["ni"] - out["ne"]) / out["ni"].replace(0.0, np.nan)
    out["electron_excess_fraction"] = (out["ne"] - out["ni"]) / out["ni"].replace(0.0, np.nan)
    return out


def build_secondary_catalog(project_root: str | Path, *, hash_roots: bool = False) -> pd.DataFrame:
    paths = ProjectPaths.from_root(project_root)
    base = paths.processed / "secondary"
    root_index = _root_index(paths)
    frames: list[pd.DataFrame] = []
    sources: list[Path] = []
    for path in sorted(base.rglob("*_secondary.csv")):
        if path.name.endswith("_IR_secondary.csv"): continue
        frame = pd.read_csv(path)
        if frame.empty: continue
        relative = str(path.relative_to(paths.root))
        frame = frame.copy()
        frame["source_csv"] = relative
        frame["source_row"] = np.arange(len(frame), dtype=int)
        frame["campaign"] = _campaign_from_path(base, path)
        frame["geometry"] = _geometry_from_path(path)
        if "file" not in frame: frame["file"] = ""
        frame["simulation_id"] = [
            _stable_id(relative, int(i), str(name))
            for i, name in zip(frame["source_row"], frame["file"], strict=False)
        ]
        root_paths, root_hashes = [], []
        for name in frame["file"].astype(str):
            matches = root_index.get(name, [])
            root_paths.append(str(matches[0].relative_to(paths.root)) if len(matches)==1 else "")
            root_hashes.append(_sha256(matches[0]) if hash_roots and len(matches)==1 else "")
        frame["root_path"] = root_paths
        frame["root_sha256"] = root_hashes
        frames.append(frame); sources.append(path)
    if not frames:
        raise FileNotFoundError(f"No *_secondary.csv tables found below {base}")
    catalog = _derive(pd.concat(frames, ignore_index=True, sort=False))
    signature = ["gas_mixture","concentration_percent","electric_field_kV_cm","pressure","gap_mm","npe"]
    for col in signature:
        if col not in catalog: catalog[col] = np.nan
    catalog["duplicate_rank"] = catalog.groupby(signature, dropna=False).cumcount()
    catalog["is_duplicate_condition"] = catalog["duplicate_rank"] > 0
    preferred = [
        "simulation_id","campaign","geometry","gas_mixture","gas1","gas2",
        "concentration_percent","concentration_fraction","pressure","electric_field_kV_cm",
        "reduced_field_kV_cm_bar","gap_mm","npe","gain","gain_std","gain_resolution",
        "ni","ni_std","ion_electron_ratio","charge_imbalance_fraction","electron_excess_fraction","alpha_eff_cm_inv","alpha_eff_over_p_cm_inv_bar",
        "n_entries","is_duplicate_condition","duplicate_rank","file","root_path","root_sha256",
        "source_csv","source_row",
    ]
    catalog = catalog[[c for c in preferred if c in catalog] + [c for c in catalog if c not in preferred]]
    out_dir = paths.secondary_cache
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "simulation_catalog.csv.gz", index=False, compression="gzip")
    return catalog
