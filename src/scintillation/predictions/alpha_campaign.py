from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.paths import ProjectPaths
from ..plotting.style import (
    BAND_ALPHA,
    CAPSIZE,
    ERRORBAR_LINEWIDTH,
    FIGSIZE_WIDE,
    LEGEND,
    LINEWIDTH_MAIN,
    MARKERSIZE,
    apply_axis_style,
    palette,
    setup_style,
)


_TARGET_COMPOSITIONS: tuple[tuple[tuple[str, float], ...], ...] = (
    (("ar", 90.0), ("cf4", 10.0)),
    (("ar", 85.0), ("cf4", 15.0)),
    (("ar", 80.0), ("cf4", 20.0)),
    (("ar", 96.0), ("ic4h10", 4.0)),
    (("ar", 95.0), ("ic4h10", 5.0)),
    (("ar", 88.0), ("ic4h10", 10.0), ("cf4", 2.0)),
    (("ar", 87.0), ("ic4h10", 10.0), ("cf4", 3.0)),
    (("ar", 78.0), ("ic4h10", 20.0), ("cf4", 2.0)),
    (("ar", 77.0), ("ic4h10", 20.0), ("cf4", 3.0)),
    (("ar", 95.0), ("ic4h10", 3.0), ("cf4", 2.0)),
)

_GAS_LABELS = {"ar": "Ar", "cf4": r"CF$_4$", "ic4h10": r"iC$_4$H$_{10}$", "iso": r"iC$_4$H$_{10}$"}
_OUTPUT_DIR = Path("secondary") / "alpha_studies" / "gap0p150_1bar_multimixture"


def _norm_gas(name: str) -> str:
    text = str(name).strip().lower()
    aliases = {
        "iso": "ic4h10",
        "isobutane": "ic4h10",
        "i-c4h10": "ic4h10",
        "ic4h10": "ic4h10",
        "cf4": "cf4",
        "ar": "ar",
    }
    return aliases.get(text, text)


def _component_key(components: list[dict[str, Any]] | tuple[tuple[str, float], ...]) -> tuple[tuple[str, float], ...]:
    if isinstance(components, tuple):
        return tuple((str(g), round(float(v), 6)) for g, v in components)
    ordered: list[tuple[str, float]] = []
    for item in components:
        gas = _norm_gas(str(item.get("gas", "")))
        frac = round(float(item.get("fraction_pct", 0.0)), 6)
        ordered.append((gas, frac))
    order = {"ar": 0, "ic4h10": 1, "cf4": 2}
    ordered.sort(key=lambda pair: (order.get(pair[0], 99), pair[0]))
    return tuple(ordered)


def _display_label(key: tuple[tuple[str, float], ...]) -> str:
    return " / ".join(f"{_GAS_LABELS.get(g, g)} {value:g}%" for g, value in key)


def _composition_token(key: tuple[tuple[str, float], ...]) -> str:
    return "__".join(f"{gas}_{value:g}".replace(".", "p") for gas, value in key)


def _load_alpha_campaign(paths: ProjectPaths) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    alpha_root = paths.raw / "garfield" / "newSimulations" / "alpha"
    rows: list[dict[str, Any]] = []
    parameters: dict[str, dict[str, Any]] = {}
    target_keys = {_component_key(item) for item in _TARGET_COMPOSITIONS}
    if not alpha_root.exists():
        return pd.DataFrame(), {}

    for json_path in sorted(alpha_root.rglob("gap_0.150mm.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        compositions = payload.get("compositions", {})
        for _, block in compositions.items():
            components = block.get("components") or []
            key = _component_key(components)
            if key not in target_keys:
                continue
            token = _composition_token(key)
            label = _display_label(key)
            params = block.get("parameters")
            if isinstance(params, dict):
                parameters[token] = params
            for point in block.get("points", []):
                rows.append({
                    "campaign_mixture": payload.get("mixture", ""),
                    "composition_token": token,
                    "composition_label": label,
                    "pressure_bar": float(point.get("pressure_bar", np.nan)),
                    "gap_mm": float(point.get("gap_mm", np.nan)),
                    "field_kV_cm": float(point.get("field_v_cm", np.nan)) / 1000.0,
                    "reduced_field_kV_cm_bar": float(point.get("field_v_cm", np.nan)) / 1000.0 / float(point.get("pressure_bar", np.nan)),
                    "gain": float(point.get("gain", np.nan)),
                    "gain_error": float(point.get("gain_error", np.nan)),
                    "alpha_effective": float(point.get("alpha_effective", np.nan)),
                    "alpha_error": float(point.get("alpha_error", np.nan)),
                    "npe": float(point.get("npe", np.nan)),
                    "root_name": Path(str(point.get("root", ""))).name,
                })
    points = pd.DataFrame(rows)
    if points.empty:
        return points, parameters
    points = points.sort_values(["composition_label", "field_kV_cm", "gain"]).reset_index(drop=True)
    out_dir = paths.secondary_cache
    out_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(out_dir / "alpha_campaign_points.csv.gz", index=False, compression="gzip")
    return points, parameters


def _merge_with_catalog(points: pd.DataFrame, catalog: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    if points.empty or catalog.empty or "file" not in catalog:
        return pd.DataFrame()
    keep = [
        "file", "npe", "Ar_1s4_1s5", "ErrAr_1s4_1s5", "Ar_1s2_1s3", "ErrAr_1s2_1s3",
        "Ar_dbleStar", "ErrAr_dbleStar", "Ar_3rd", "ErrAr_3rd",
    ]
    available = [column for column in keep if column in catalog.columns]
    merged = points.merge(catalog[available].rename(columns={"file": "root_name"}), how="left", on="root_name", suffixes=("", "_catalog"))
    if merged.empty:
        return merged
    nscale = pd.to_numeric(merged.get("npe_catalog", merged.get("npe", np.nan)), errors="coerce").replace(0.0, np.nan)
    vuv_terms = [pd.to_numeric(merged.get(col, 0.0), errors="coerce").fillna(0.0) for col in ("Ar_1s4_1s5", "Ar_1s2_1s3", "Ar_dbleStar") if col in merged]
    vuv_errs = [pd.to_numeric(merged.get(col, 0.0), errors="coerce").fillna(0.0) for col in ("ErrAr_1s4_1s5", "ErrAr_1s2_1s3", "ErrAr_dbleStar") if col in merged]
    uv = pd.to_numeric(merged.get("Ar_3rd", np.nan), errors="coerce")
    uv_err = pd.to_numeric(merged.get("ErrAr_3rd", np.nan), errors="coerce")
    merged["vuv_per_primary"] = sum(vuv_terms) / nscale if vuv_terms else np.nan
    merged["vuv_per_primary_err"] = np.sqrt(sum(term**2 for term in vuv_errs)) / nscale if vuv_errs else np.nan
    merged["uv_per_primary"] = uv / nscale
    merged["uv_per_primary_err"] = uv_err / nscale
    out_dir = paths.secondary_cache
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "alpha_campaign_merged.csv.gz", index=False, compression="gzip")
    return merged


def _gain_from_parameters(params: dict[str, Any], reduced_field: np.ndarray, pressure_bar: float, gap_mm: float) -> np.ndarray:
    A = float(params["A"])
    B = float(params["B"])
    m = float(params["m"])
    n = float(params["n"])
    alpha_over_p = A * np.power(reduced_field, m) * np.exp(-np.power(B / np.maximum(reduced_field, 1.0e-12), n))
    alpha = pressure_bar * alpha_over_p
    return np.exp(alpha * (gap_mm / 10.0))


def _sample_gain_band(params: dict[str, Any], reduced_field: np.ndarray, pressure_bar: float, gap_mm: float, *, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray] | None:
    covariance = params.get("covariance")
    if covariance is None:
        return None
    try:
        mean = np.asarray([params["A"], params["B"], params["m"], params["n"]], dtype=float)
        cov = np.asarray(covariance, dtype=float)
        draws = np.random.default_rng(12345).multivariate_normal(mean, cov, size=n_samples)
    except Exception:
        return None
    valid = np.isfinite(draws).all(axis=1)
    valid &= draws[:, 0] > 0.0
    valid &= draws[:, 1] >= 0.0
    valid &= draws[:, 2] > 0.0
    valid &= draws[:, 3] > 0.0
    draws = draws[valid]
    if draws.size == 0:
        return None
    curves = []
    for A, B, m, n in draws:
        alpha_over_p = A * np.power(reduced_field, m) * np.exp(-np.power(B / np.maximum(reduced_field, 1.0e-12), n))
        gain = np.exp((pressure_bar * alpha_over_p) * (gap_mm / 10.0))
        if np.isfinite(gain).all():
            curves.append(gain)
    if not curves:
        return None
    stack = np.asarray(curves, dtype=float)
    return np.percentile(stack, 16.0, axis=0), np.percentile(stack, 84.0, axis=0)


def _axis_finish(ax, *, xlabel: str, ylabel: str, title: str, xscale: str = "linear", yscale: str = "linear") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    apply_axis_style(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **LEGEND.as_kwargs(loc="best", fontsize=8.9, title_fontsize=8.9, ncol=1))


def _plot_gain(points: pd.DataFrame, params_map: dict[str, dict[str, Any]], out_dir: Path, colors: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for label, group in points.groupby("composition_label", sort=True):
        token = group["composition_token"].iloc[0]
        group = group.sort_values("field_kV_cm")
        color = colors[label]
        ax.errorbar(group["field_kV_cm"], group["gain"], yerr=group["gain_error"], fmt="o",
                    color=color, ecolor=color, markersize=MARKERSIZE, linewidth=ERRORBAR_LINEWIDTH,
                    elinewidth=ERRORBAR_LINEWIDTH, capsize=CAPSIZE, label=label)
        ax.plot(group["field_kV_cm"], group["gain"], color=color, linewidth=1.3, alpha=0.75)
        params = params_map.get(token)
        if isinstance(params, dict):
            rmin, rmax = params.get("valid_reduced_field", [group["field_kV_cm"].min(), group["field_kV_cm"].max()])
            reduced = np.linspace(float(rmin), float(rmax), 240)
            gain_curve = _gain_from_parameters(params, reduced, pressure_bar=1.0, gap_mm=0.15)
            ax.plot(reduced, gain_curve, color=color, linewidth=LINEWIDTH_MAIN, alpha=0.60)
            band = _sample_gain_band(params, reduced, pressure_bar=1.0, gap_mm=0.15)
            if band is not None:
                low, high = band
                ax.fill_between(reduced, low, high, color=color, alpha=max(0.12, 0.75 * BAND_ALPHA))
    _axis_finish(ax, xlabel=r"Electric field [kV cm$^{-1}$]", ylabel="Gain", title="Gain vs electric field (1 bar, 150 μm gap)", yscale="log")
    fig.savefig(out_dir / "gain_vs_efield.pdf")
    plt.close(fig)


def _plot_emission(frame: pd.DataFrame, *, ycol: str, yerr: str, xcol: str, out_path: Path, title: str, ylabel: str, colors: dict[str, Any], xscale: str = "linear") -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for label, group in frame.groupby("composition_label", sort=True):
        group = group.sort_values(xcol)
        color = colors[label]
        x = pd.to_numeric(group[xcol], errors="coerce")
        y = pd.to_numeric(group[ycol], errors="coerce")
        ye = pd.to_numeric(group[yerr], errors="coerce") if yerr in group else pd.Series(np.nan, index=group.index)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        x = x[valid]
        y = y[valid]
        ye = ye[valid]
        ax.plot(x, y, color=color, linewidth=1.4, alpha=0.80)
        ax.errorbar(x, y, yerr=ye if np.isfinite(ye).any() else None, fmt="o", color=color, ecolor=color,
                    markersize=MARKERSIZE, linewidth=ERRORBAR_LINEWIDTH, elinewidth=ERRORBAR_LINEWIDTH,
                    capsize=CAPSIZE, label=label)
    _axis_finish(ax, xlabel=(r"Electric field [kV cm$^{-1}$]" if xcol == "field_kV_cm" else "Gain"),
                 ylabel=ylabel, title=title, xscale=xscale, yscale="log")
    fig.savefig(out_path)
    plt.close(fig)


def render_alpha_campaign_plots(project_root: str | Path, catalog: pd.DataFrame) -> None:
    paths = ProjectPaths.from_root(project_root)
    points, params_map = _load_alpha_campaign(paths)
    if points.empty:
        return
    filtered = points.loc[(points["pressure_bar"].round(6) == 1.0) & (points["gap_mm"].round(6) == 0.15)].copy()
    if filtered.empty:
        return
    merged = _merge_with_catalog(filtered, catalog, paths)
    setup_style(grid=False, use_latex=False, context="single")
    out_dir = paths.figures / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(pd.unique(filtered["composition_label"]))
    colors = {label: color for label, color in zip(labels, palette(len(labels), start=0.04, stop=0.96), strict=False)}
    _plot_gain(filtered, params_map, out_dir, colors)
    if merged.empty:
        return
    _plot_emission(merged, ycol="vuv_per_primary", yerr="vuv_per_primary_err", xcol="field_kV_cm",
                   out_path=out_dir / "vuv_vs_efield.pdf",
                   title="VUV vs electric field (1 bar, 150 μm gap)",
                   ylabel=r"VUV signal [arb. u. / primary e$^{-}$]",
                   colors=colors, xscale="linear")
    _plot_emission(merged, ycol="vuv_per_primary", yerr="vuv_per_primary_err", xcol="gain",
                   out_path=out_dir / "vuv_vs_gain.pdf",
                   title="VUV vs gain (1 bar, 150 μm gap)",
                   ylabel=r"VUV signal [arb. u. / primary e$^{-}$]",
                   colors=colors, xscale="log")
    _plot_emission(merged, ycol="uv_per_primary", yerr="uv_per_primary_err", xcol="field_kV_cm",
                   out_path=out_dir / "uv_vs_efield.pdf",
                   title="UV vs electric field (1 bar, 150 μm gap)",
                   ylabel=r"UV signal [arb. u. / primary e$^{-}$]",
                   colors=colors, xscale="linear")
    _plot_emission(merged, ycol="uv_per_primary", yerr="uv_per_primary_err", xcol="gain",
                   out_path=out_dir / "uv_vs_gain.pdf",
                   title="UV vs gain (1 bar, 150 μm gap)",
                   ylabel=r"UV signal [arb. u. / primary e$^{-}$]",
                   colors=colors, xscale="log")
