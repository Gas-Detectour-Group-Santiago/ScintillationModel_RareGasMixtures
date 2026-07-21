#!/usr/bin/env python3
"""Additional low-pressure predictions from ``ArJoint_IR_primary``.

Outputs are kept separate from the legacy independent-fit products.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import setup_style
from scintillation.fitting.toy_cache import load_toys
from scintillation.plotting.recipe_config import (
    as_bool,
    as_float,
    as_int,
    as_limits,
    as_text,
    grid_from_row,
    output_path,
    primary_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for folder in ("models", "primary_fits"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

from ArJoint_infrarred import IR_LINES, theory_yield_joint, theory_yield_total

setup_style(grid=False, use_latex=False, context="single")

DATA_DIR = PROJECT_ROOT / "data"
FIT_NAME = "ArJoint_IR_primary"
PRESSURES_MBAR = np.asarray([0.1, 1.0, 10.0, 50.0, 100.0, 1000.0], dtype=float)
Y_LIM = (500.0, 30000.0)


def read_parameter_products() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    central_path = DATA_DIR / "Parameters" / f"{FIT_NAME}.csv"
    if not central_path.exists():
        raise FileNotFoundError(
            f"Missing {central_path}. Run: python primary_fits/ArJoint_IR_fit.py"
        )
    central = pd.read_csv(central_path)["value"].to_numpy(dtype=float)

    names = pd.read_csv(central_path)["name"].astype(str).tolist()
    return (
        central,
        load_toys(DATA_DIR / "FitResults", FIT_NAME, "stat", names),
        load_toys(DATA_DIR / "FitResults", FIT_NAME, "syst", names),
    )


def reference_norm() -> float:
    table = pd.read_csv(DATA_DIR / "Parameters" / "ArCF4_primary.csv")
    rows = table.loc[table["name"].astype(str) == "Nnorm", "value"]
    if rows.empty:
        raise KeyError("Nnorm missing from data/Parameters/ArCF4_primary.csv")
    return float(rows.iloc[0])


def degrad_tables() -> dict[str, pd.DataFrame]:
    return {
        "ArCF4": pd.read_csv(DATA_DIR / "Primary_DegradData" / "ArCF4_IR.csv"),
        "ArN2": pd.read_csv(DATA_DIR / "Primary_DegradData" / "ArN2_IR.csv"),
    }


def to_ph_mev(raw, norm: float):
    return np.asarray(raw, dtype=float) * 1000.0 / float(norm)


def quantile_errors(samples: np.ndarray, central: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    central = np.asarray(central, dtype=float)
    if samples.size == 0:
        return np.zeros_like(central), np.zeros_like(central)
    q16, q84 = np.nanpercentile(samples, [16.0, 84.0], axis=0)
    return np.maximum(central - q16, 0.0), np.maximum(q84 - central, 0.0)


def evaluate_total(params: np.ndarray, degrad: pd.DataFrame, mixture: str, f, p_bar: float, norm: float):
    return to_ph_mev(theory_yield_total(params, degrad, f, p_bar, mixture=mixture), norm)


def evaluate_toys(toys: np.ndarray, degrad: pd.DataFrame, mixture: str, f, p_bar: float, norm: float) -> np.ndarray:
    if toys.size == 0:
        return np.empty((0, np.size(f)), dtype=float)
    return np.vstack([evaluate_total(row, degrad, mixture, f, p_bar, norm) for row in toys])


def build_curve_frame(
    central: np.ndarray,
    stat_toys: np.ndarray,
    syst_toys: np.ndarray,
    degrad: pd.DataFrame,
    mixture: str,
    concentration_fraction: np.ndarray,
    pressure_mbar: float,
    norm: float,
) -> pd.DataFrame:
    p_bar = pressure_mbar / 1000.0
    value = evaluate_total(central, degrad, mixture, concentration_fraction, p_bar, norm)
    stat_samples = evaluate_toys(stat_toys, degrad, mixture, concentration_fraction, p_bar, norm)
    syst_samples = evaluate_toys(syst_toys, degrad, mixture, concentration_fraction, p_bar, norm)
    stat_minus, stat_plus = quantile_errors(stat_samples, value)
    syst_minus, syst_plus = quantile_errors(syst_samples, value)
    total_minus = np.sqrt(stat_minus**2 + syst_minus**2)
    total_plus = np.sqrt(stat_plus**2 + syst_plus**2)
    return pd.DataFrame(
        {
            "concentration_fraction": concentration_fraction,
            "concentration_percent": concentration_fraction * 100.0,
            "pressure_mbar": pressure_mbar,
            "value": value,
            "stat_minus": stat_minus,
            "stat_plus": stat_plus,
            "syst_minus": syst_minus,
            "syst_plus": syst_plus,
            "total_minus": total_minus,
            "total_plus": total_plus,
        }
    )


def _joint_plot_rows(plot_type: str) -> pd.DataFrame:
    rows = primary_rows(PROJECT_ROOT, "joint_ir")
    if rows.empty:
        return rows
    return rows.loc[rows["plot_type"] == plot_type].copy()


def export_overlay_products(
    central: np.ndarray,
    stat_toys: np.ndarray,
    syst_toys: np.ndarray,
    degrad: dict[str, pd.DataFrame],
    norm: float,
) -> None:
    band_dir = DATA_DIR / "Predictions" / "Bands" / "ArJoint_IR"
    band_dir.mkdir(parents=True, exist_ok=True)
    recipes = _joint_plot_rows("joint_multiband")
    if recipes.empty:
        raise RuntimeError("No active joint_multiband rows in config/plots/primary.csv")
    cmap = plt.get_cmap("viridis")

    for plot_id, plot_rows in recipes.groupby("plot_id", sort=False):
        first = plot_rows.iloc[0]
        mixture = as_text(first.get("mixture"))
        if mixture not in degrad:
            raise KeyError(f"Unknown joint-IR mixture {mixture!r} in {plot_id}")
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        all_frames: list[pd.DataFrame] = []
        colors = cmap(np.linspace(0.10, 0.90, max(len(plot_rows), 2)))
        for color, (_, row) in zip(colors, plot_rows.iterrows()):
            pressure_bar = float(as_float(row.get("pressure_bar"), 1.0))
            pressure_mbar = 1000.0 * pressure_bar
            f_grid = grid_from_row(row)
            frame = build_curve_frame(
                central, stat_toys, syst_toys, degrad[mixture], mixture, f_grid, pressure_mbar, norm
            )
            frame["plot_id"] = str(plot_id)
            frame["series_id"] = as_text(row.get("series_id"))
            all_frames.append(frame)
            x = frame["concentration_percent"].to_numpy(dtype=float)
            y = frame["value"].to_numpy(dtype=float)
            ax.plot(
                x, y, color=color,
                lw=float(as_float(row.get("linewidth"), 1.7)),
                linestyle=as_text(row.get("linestyle"), "-"),
                marker=as_text(row.get("marker")) or None,
                label=as_text(row.get("label"), f"{pressure_mbar:g} mbar"),
            )
            band_mode = as_text(row.get("bands"), "total")
            show_band = as_bool(row.get("show_band"), band_mode not in {"", "none"})
            if show_band and band_mode not in {"", "none"}:
                if band_mode == "stat":
                    minus = frame["stat_minus"].to_numpy(dtype=float)
                    plus = frame["stat_plus"].to_numpy(dtype=float)
                elif band_mode == "syst":
                    minus = frame["syst_minus"].to_numpy(dtype=float)
                    plus = frame["syst_plus"].to_numpy(dtype=float)
                else:
                    minus = frame["total_minus"].to_numpy(dtype=float)
                    plus = frame["total_plus"].to_numpy(dtype=float)
                ax.fill_between(x, np.maximum(y - minus, 1e-30), y + plus, color=color, alpha=0.12, linewidth=0)

        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(band_dir / f"{plot_id}.csv", index=False)
        ax.set_xscale(as_text(first.get("xscale"), "log"))
        ax.set_yscale(as_text(first.get("yscale"), "log"))
        xlim = as_limits(first, "x"); ylim = as_limits(first, "y")
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        ax.set_xlabel(as_text(first.get("xlabel")))
        ax.set_ylabel(as_text(first.get("ylabel")))
        ax.set_title(as_text(first.get("title")))
        ax.legend(
            ncol=int(as_int(first.get("legend_ncol"), 2) or 2),
            fontsize=as_float(first.get("legend_fontsize"), 8.0),
            loc=as_text(first.get("legend_loc"), "best"),
        )
        fig.tight_layout()
        destination = output_path(first, PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)


def pure_ar_samples(
    params_rows: np.ndarray,
    degrad: dict[str, pd.DataFrame],
    pressure_bar: float,
    norm: float,
) -> np.ndarray:
    if params_rows.size == 0:
        return np.empty((0, 3), dtype=float)
    rows = []
    for params in params_rows:
        y_cf4 = float(evaluate_total(params, degrad["ArCF4"], "ArCF4", np.asarray([0.0]), pressure_bar, norm)[0])
        y_n2 = float(evaluate_total(params, degrad["ArN2"], "ArN2", np.asarray([0.0]), pressure_bar, norm)[0])
        rows.append((y_cf4, y_n2, 0.5 * (y_cf4 + y_n2)))
    return np.asarray(rows, dtype=float)


def export_pure_ar_table_and_plot(
    central: np.ndarray,
    stat_toys: np.ndarray,
    syst_toys: np.ndarray,
    degrad: dict[str, pd.DataFrame],
    norm: float,
) -> None:
    rows: list[dict] = []
    for pressure_mbar in PRESSURES_MBAR:
        p_bar = pressure_mbar / 1000.0
        y_cf4 = float(evaluate_total(central, degrad["ArCF4"], "ArCF4", np.asarray([0.0]), p_bar, norm)[0])
        y_n2 = float(evaluate_total(central, degrad["ArN2"], "ArN2", np.asarray([0.0]), p_bar, norm)[0])
        mean = 0.5 * (y_cf4 + y_n2)
        half_spread = 0.5 * abs(y_cf4 - y_n2)
        stat = pure_ar_samples(stat_toys, degrad, p_bar, norm)
        syst = pure_ar_samples(syst_toys, degrad, p_bar, norm)
        stat_minus, stat_plus = quantile_errors(stat[:, 2] if stat.size else np.empty((0,)), np.asarray(mean))
        syst_minus, syst_plus = quantile_errors(syst[:, 2] if syst.size else np.empty((0,)), np.asarray(mean))
        sm, sp = float(stat_minus), float(stat_plus)
        ym, yp = float(syst_minus), float(syst_plus)
        rows.append(
            {
                "pressure_mbar": pressure_mbar,
                "arcf4_degrad_value": y_cf4,
                "arn2_degrad_value": y_n2,
                "value_mean": mean,
                "population_half_spread": half_spread,
                "population_half_spread_percent": 100.0 * half_spread / mean,
                "stat_minus": sm,
                "stat_plus": sp,
                "syst_minus": ym,
                "syst_plus": yp,
                "total_minus": float(np.hypot(sm, ym)),
                "total_plus": float(np.hypot(sp, yp)),
            }
        )
    frame = pd.DataFrame(rows)
    pred_dir = DATA_DIR / "Predictions"
    table_dir = DATA_DIR / "Tables"
    plot_dir = PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "joint_ir"
    pred_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(pred_dir / f"{FIT_NAME}_low_pressure_pure_ar.csv", index=False)

    def num(v: float) -> str:
        return rf"\num{{{float(v):.3g}}}"

    def asym(m: float, p: float) -> str:
        if m == 0 and p == 0:
            return r"--"
        return rf"$^{{+{num(p)}}}_{{-{num(m)}}}$"

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Predicción IR del argón puro obtenida con el ajuste conjunto Ar--CF$_4$/Ar--N$_2$. Las dos primeras columnas emplean las poblaciones Degrad de cada simulación; el valor recomendado es su media.}",
        r"\label{tab:ArJoint_IR_low_pressure_pure_ar}",
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        r"$p$ [mbar] & Degrad Ar--CF$_4$ & Degrad Ar--N$_2$ & Media & $\Delta_{\rm pop.}$ & Total \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        tex.append(
            f"{row['pressure_mbar']:g} & {num(row['arcf4_degrad_value'])} & {num(row['arn2_degrad_value'])} & "
            f"{num(row['value_mean'])} & {num(row['population_half_spread'])} & "
            f"{asym(row['total_minus'], row['total_plus'])} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (table_dir / f"{FIT_NAME}_low_pressure_pure_ar.tex").write_text("\n".join(tex), encoding="utf-8")

    plot_recipes = _joint_plot_rows("joint_pure_ar")
    if plot_recipes.empty:
        raise RuntimeError("No active joint_pure_ar rows in config/plots/primary.csv")
    for _plot_id, rows_for_plot in plot_recipes.groupby("plot_id", sort=False):
        first = rows_for_plot.iloc[0]
        x = frame["pressure_mbar"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        component_columns = {
            "arcf4_degrad": "arcf4_degrad_value",
            "arn2_degrad": "arn2_degrad_value",
            "mean": "value_mean",
        }
        for _, recipe in rows_for_plot.iterrows():
            component = as_text(recipe.get("component"))
            if component not in component_columns:
                raise KeyError(f"Unknown pure-Ar component {component!r}")
            y = frame[component_columns[component]].to_numpy(dtype=float)
            ax.plot(
                x, y,
                linestyle=as_text(recipe.get("linestyle"), "-"),
                linewidth=float(as_float(recipe.get("linewidth"), 1.5)),
                marker=as_text(recipe.get("marker")) or None,
                markersize=4,
                label=as_text(recipe.get("label"), component),
            )
            bands = as_text(recipe.get("bands"), "none")
            if component == "mean" and as_bool(recipe.get("show_band"), bands != "none") and bands != "none":
                if bands == "stat":
                    minus = frame["stat_minus"].to_numpy(dtype=float); plus = frame["stat_plus"].to_numpy(dtype=float)
                elif bands == "syst":
                    minus = frame["syst_minus"].to_numpy(dtype=float); plus = frame["syst_plus"].to_numpy(dtype=float)
                else:
                    minus = frame["total_minus"].to_numpy(dtype=float); plus = frame["total_plus"].to_numpy(dtype=float)
                ax.fill_between(x, np.maximum(y - minus, 1e-30), y + plus, alpha=0.18, linewidth=0)
        ax.set_xscale(as_text(first.get("xscale"), "log"))
        ax.set_yscale(as_text(first.get("yscale"), "log"))
        xlim = as_limits(first, "x"); ylim = as_limits(first, "y")
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        ax.set_xlabel(as_text(first.get("xlabel")))
        ax.set_ylabel(as_text(first.get("ylabel")))
        ax.set_title(as_text(first.get("title")))
        ax.legend(
            fontsize=as_float(first.get("legend_fontsize"), 8.0),
            ncol=int(as_int(first.get("legend_ncol"), 1) or 1),
            loc=as_text(first.get("legend_loc"), "best"),
        )
        fig.tight_layout()
        destination = output_path(first, PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)


def export_component_table(central: np.ndarray, degrad: dict[str, pd.DataFrame], norm: float) -> None:
    rows = []
    for pressure_mbar in PRESSURES_MBAR:
        p_bar = pressure_mbar / 1000.0
        for line in IR_LINES:
            vals = []
            for mixture in ("ArCF4", "ArN2"):
                raw = theory_yield_joint(
                    central,
                    degrad[mixture],
                    np.asarray([0.0]),
                    p_bar,
                    mixture=mixture,
                    line=line,
                )
                vals.append(float(to_ph_mev(raw, norm)[0]))
            rows.append(
                {
                    "pressure_mbar": pressure_mbar,
                    "line_nm": 764 if line == "763" else int(line),
                    "arcf4_degrad_value": vals[0],
                    "arn2_degrad_value": vals[1],
                    "value_mean": 0.5 * sum(vals),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "Predictions" / f"{FIT_NAME}_low_pressure_components.csv", index=False)


def main() -> None:
    central, stat_toys, syst_toys = read_parameter_products()
    norm = reference_norm()
    degrad = degrad_tables()
    export_overlay_products(central, stat_toys, syst_toys, degrad, norm)
    export_pure_ar_table_and_plot(central, stat_toys, syst_toys, degrad, norm)
    export_component_table(central, degrad, norm)
    print("[ArJoint IR predictions] products written under data/Predictions, data/Tables and primary_predictions/plots/primary_bands/joint_ir")


if __name__ == "__main__":
    main()
