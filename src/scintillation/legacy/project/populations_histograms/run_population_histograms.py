#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Energy histograms from Garfield++/Magboltz level catalogues.

Default input
-------------
The script reads level catalogues from::

    data/Secondary_GarfieldData/levels/

with the format used by the existing ``*_level_data.csv`` files::

    level,gas,state_name,type,energy_eV,n_collisions

It separates the information by gas (Ar, CF4, N2, He) using the internal
``gas`` column, not the filename.  By default, duplicate levels appearing in
more than one mixture file are removed.

Important
---------
These files usually describe available levels/channels, not simulated event
populations.  If ``n_collisions`` is zero everywhere, the histogram counts one
entry per level/channel.  If ``n_collisions`` contains positive values, the
script can use those as weights with ``--weight-mode auto`` or
``--weight-mode collisions``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from plot_style import FIGSIZE_HISTOGRAM, LEGEND, setup_style
except Exception:  # pragma: no cover - fallback for standalone use
    FIGSIZE_HISTOGRAM = (7.8, 5.0)

    class _LegendFallback:
        def as_kwargs(self, **overrides: object) -> dict[str, object]:
            values: dict[str, object] = {
                "fontsize": 11,
                "frameon": False,
                "handlelength": 2.2,
                "borderaxespad": 0.35,
            }
            values.update(overrides)
            return values

    LEGEND = _LegendFallback()

    def setup_style(*, grid: bool = False, use_latex: bool = False) -> None:
        del use_latex
        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "dejavuserif",
                "axes.grid": grid,
                "axes.titlesize": 15,
                "axes.labelsize": 15,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 11,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.bbox": "tight",
            }
        )


DEFAULT_INPUT_DIR = ROOT / "data" / "Secondary_GarfieldData" / "levels"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "populations" / "plots"
DEFAULT_BIN_WIDTH_EV = 0.5
# Main TFM figures are restricted to the physically useful/visible region.
DEFAULT_MAX_ENERGY_EV = 50.0

GAS_ORDER = ["Ar", "CF4", "He", "N2"]

GROUP_ORDER = [
    "Elastic / zero-loss",
    "Attachment",
    "Ionisation",
    "Dissociative / double ionisation",
    "Atomic excitation",
    "Electronic excitation",
    "Vibrational excitation",
    "Rotational excitation",
    "Neutral dissociation",
    "Superelastic",
    "Other",
]

GROUP_COLORS = {
    "Elastic / zero-loss": "#9E9E9E",
    "Attachment": "#7B3294",
    "Ionisation": "#B22222",
    "Dissociative / double ionisation": "#D95F02",
    "Atomic excitation": "#1F77B4",
    "Electronic excitation": "#4C78A8",
    "Vibrational excitation": "#2CA02C",
    "Rotational excitation": "#17BECF",
    "Neutral dissociation": "#FF7F0E",
    "Superelastic": "#666666",
    "Other": "#C7C7C7",
}

GAS_LABELS = {
    "Ar": r"Ar",
    "CF4": r"CF$_4$",
    "He": r"He",
    "N2": r"N$_2$",
}

REQUIRED_COLUMNS = {"gas", "state_name", "type", "energy_eV"}


def normalise_gas(value: object) -> str:
    text = str(value).strip().upper().replace(" ", "")
    if text in {"AR", "ARGON"}:
        return "Ar"
    if text in {"CF4", "C-F4"}:
        return "CF4"
    if text in {"HE", "HELIUM", "HELIUM4"}:
        return "He"
    if text in {"N2", "NITROGEN"}:
        return "N2"
    return str(value).strip()


def classify_level(gas: str, level_type: object, state_name: object, energy_eV: float) -> str:
    typ = str(level_type).strip().lower()
    name = str(state_name).upper()

    if typ == "attachment" or "ATTACH" in name:
        return "Attachment"
    if typ == "elastic" or abs(float(energy_eV)) < 1e-12 or "ELASTIC" in name:
        return "Elastic / zero-loss"
    if typ == "superelastic" or float(energy_eV) < 0:
        return "Superelastic"

    if "NEUTRAL DISS" in name:
        return "Neutral dissociation"
    if "VIB" in name or "VIBRATION" in name:
        return "Vibrational excitation"
    if "ROTATION" in name or re.search(r"\bROT\b", name):
        return "Rotational excitation"

    if typ == "ionisation" or "IONIS" in name or re.search(r"\bION\b", name):
        if "DOUBLE" in name or "DISSOC ION" in name or "++" in name or "," in name:
            return "Dissociative / double ionisation"
        return "Ionisation"

    if "DISS" in name:
        return "Neutral dissociation"

    if typ == "excitation":
        if gas in {"Ar", "He"}:
            return "Atomic excitation"
        return "Electronic excitation"

    if typ == "inelastic":
        if gas == "CF4" and "NEUTRAL" in name:
            return "Neutral dissociation"
        return "Electronic excitation"

    return "Other"


def discover_level_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = sorted(input_dir.rglob("*_level_data.csv"))
    if files:
        return files
    return sorted(input_dir.rglob("*.csv"))


def read_level_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["source_file"] = path.name
    out["gas"] = out["gas"].map(normalise_gas)
    out["energy_eV"] = pd.to_numeric(out["energy_eV"], errors="coerce")
    out = out.dropna(subset=["energy_eV"])

    if "n_collisions" not in out.columns:
        out["n_collisions"] = 0.0
    out["n_collisions"] = pd.to_numeric(out["n_collisions"], errors="coerce").fillna(0.0)

    out["group"] = [
        classify_level(gas, typ, name, energy)
        for gas, typ, name, energy in zip(
            out["gas"], out["type"], out["state_name"], out["energy_eV"]
        )
    ]
    return out


def load_all_levels(input_dir: Path, deduplicate: bool = True) -> pd.DataFrame:
    files = discover_level_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No level CSV files found in {input_dir}")

    tables = []
    for path in files:
        try:
            tables.append(read_level_file(path))
        except Exception as exc:
            print(f"warning: skipped {path}: {exc}")

    if not tables:
        raise RuntimeError("No valid level tables could be loaded.")

    levels = pd.concat(tables, ignore_index=True)
    levels = levels[levels["gas"].isin(GAS_ORDER)].copy()

    if deduplicate:
        key_cols = ["gas", "state_name", "type", "energy_eV", "group"]
        levels = levels.drop_duplicates(subset=key_cols).copy()

    return levels


def choose_weights(rows: pd.DataFrame, weight_mode: str) -> tuple[np.ndarray, str]:
    collisions = rows["n_collisions"].to_numpy(dtype=float)

    if weight_mode == "levels":
        return np.ones(len(rows), dtype=float), "Number of listed channels / levels"
    if weight_mode == "collisions":
        return collisions, "Number of Garfield collisions"
    if weight_mode == "auto":
        if np.nanmax(collisions) > 0:
            return collisions, "Number of Garfield collisions"
        return np.ones(len(rows), dtype=float), "Number of listed channels / levels"

    raise ValueError(f"unknown weight mode: {weight_mode}")


def make_bins(energies: Iterable[float], bin_width_ev: float, max_energy: float | None) -> np.ndarray:
    energies = np.asarray(list(energies), dtype=float)
    if energies.size == 0:
        return np.array([0.0, bin_width_ev])
    upper = float(max_energy) if max_energy is not None else float(np.nanmax(energies))
    upper = max(upper, bin_width_ev)
    return np.arange(0.0, upper + bin_width_ev, bin_width_ev)


def plot_gas_histogram(
    gas: str,
    rows: pd.DataFrame,
    output_dir: Path,
    bin_width_ev: float,
    weight_mode: str,
    max_energy: float | None,
    include_superelastic: bool,
) -> Path | None:
    rows = rows.copy()
    if not include_superelastic:
        rows = rows[rows["energy_eV"] >= 0].copy()
    rows = rows[rows["gas"] == gas].copy()
    if rows.empty:
        return None

    n_available = len(rows)
    if max_energy is not None:
        rows = rows[rows["energy_eV"] <= float(max_energy)].copy()
    n_shown = len(rows)
    if rows.empty:
        return None

    weights, ylabel = choose_weights(rows, weight_mode)
    rows["weight"] = weights
    rows = rows[rows["weight"] > 0].copy()
    if rows.empty:
        return None

    bins = make_bins(rows["energy_eV"], bin_width_ev=bin_width_ev, max_energy=max_energy)

    setup_style(grid=False, use_latex=False)
    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)

    bottom = np.zeros(len(bins) - 1, dtype=float)
    present_groups = [group for group in GROUP_ORDER if group in set(rows["group"])]

    for group in present_groups:
        sub = rows[rows["group"] == group]
        hist, _ = np.histogram(sub["energy_eV"], bins=bins, weights=sub["weight"])
        if np.all(hist == 0):
            continue
        ax.bar(
            bins[:-1],
            hist,
            bottom=bottom,
            width=np.diff(bins),
            align="edge",
            label=group,
            color=GROUP_COLORS.get(group, "#C7C7C7"),
            edgecolor="none",
            alpha=0.95,
        )
        bottom += hist

    gas_label = GAS_LABELS.get(gas, gas)
    ax.set_xlabel(r"Energy loss / excitation energy [eV]")
    ax.set_ylabel(ylabel)
    if max_energy is not None:
        ax.set_title(f"{gas_label} Garfield++ level catalogue, 0--{max_energy:g} eV")
        ax.set_xlim(0.0, float(max_energy))
    else:
        ax.set_title(f"{gas_label} Garfield++ level catalogue")
        ax.set_xlim(bins[0], bins[-1])

    if n_shown < n_available:
        ax.text(
            0.02,
            0.96,
            f"showing {n_shown}/{n_available} levels",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
    ax.legend(loc="upper right", **LEGEND.as_kwargs())

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{gas}_level_population_histogram.pdf"
    fig.savefig(output)
    plt.close(fig)
    return output


def print_summary(levels: pd.DataFrame) -> None:
    summary = (
        levels.groupby(["gas", "group"], observed=True)
        .size()
        .rename("n_levels")
        .reset_index()
        .sort_values(["gas", "group"])
    )
    print("\nLoaded level/channel summary:")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH_EV)
    parser.add_argument(
        "--weight-mode",
        choices=("auto", "levels", "collisions"),
        default="auto",
        help="auto uses n_collisions only if any positive values exist; otherwise counts levels.",
    )
    parser.add_argument("--max-energy", type=float, default=DEFAULT_MAX_ENERGY_EV, help="Upper x-axis limit in eV. Default: 50 eV. Use --max-energy 0 to auto-scale to all energies.")
    parser.add_argument("--include-superelastic", action="store_true")
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.max_energy is not None and args.max_energy <= 0:
        args.max_energy = None

    levels = load_all_levels(args.input_dir, deduplicate=not args.no_deduplicate)

    if args.summary:
        print_summary(levels)

    written = []
    for gas in GAS_ORDER:
        output = plot_gas_histogram(
            gas=gas,
            rows=levels,
            output_dir=args.output_dir,
            bin_width_ev=args.bin_width,
            weight_mode=args.weight_mode,
            max_energy=args.max_energy,
            include_superelastic=args.include_superelastic,
        )
        if output is not None:
            written.append(output)
            print(f"wrote {output}")

    if not written:
        raise SystemExit("No histograms were written. Check input files and weight mode.")


if __name__ == "__main__":
    main()