"""
Utilities for statistically and systematically separated fit bands.

Conventions used here
---------------------
* Statistical toys: each experimental point gets an independent Gaussian pull.
* Systematic toys: one Gaussian nuisance is shared by a physics family
  (UV/VIS channel, IR line, etc.).  This keeps the systematic correlated and
  avoids treating it as another point-to-point statistical fluctuation.
* Central/optimal curves are always evaluated with the parameter CSV produced
  by the primary fit scripts in data/Parameters/.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd


STAT_PERCENTILES = (16.0, 84.0)


def project_root_from_file(file: str | Path, levels_up: int = 1) -> Path:
    path = Path(file).resolve()
    root = path.parent
    for _ in range(levels_up):
        root = root.parent
    return root


def configure_matplotlib(plt, *, no_grid: bool = True) -> None:
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass

    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 4.3),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlepad": 7,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    if no_grid:
        plt.rcParams["axes.grid"] = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pressure_cols(pressures: Iterable[float]) -> tuple[list[str], list[str]]:
    bars = [f"{float(p):.1f}bar" for p in pressures]
    errs = [f"Err {c}" for c in bars]
    return bars, errs


def infer_pressure_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    bars = []
    errs = []
    for col in df.columns:
        col_s = str(col)
        if col_s.endswith("bar") and not col_s.startswith("Err "):
            err = f"Err {col_s}"
            if err in df.columns:
                bars.append(col_s)
                errs.append(err)
    return bars, errs


def read_primary_parameters(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    if "parameter" not in df.columns:
        raise ValueError(f"{path} no contiene una columna 'parameter'.")
    return df["parameter"].to_numpy(dtype=float)


def parameter_index_names(path: str | Path) -> list[str]:
    return list(pd.read_csv(path, index_col=0).index.astype(str))


def result_like(x: np.ndarray, perr: np.ndarray | None = None) -> SimpleNamespace:
    x = np.asarray(x, dtype=float)
    if perr is None:
        perr = np.full_like(x, np.nan, dtype=float)
    else:
        perr = np.asarray(perr, dtype=float)
    return SimpleNamespace(x=x, perr=perr, pcov=np.diag(np.nan_to_num(perr, nan=0.0) ** 2))


def _safe_percentiles(samples: np.ndarray, nominal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nominal = np.asarray(nominal, dtype=float)
    samples = np.asarray(samples, dtype=float)

    nan = np.full_like(nominal, np.nan, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return nan, nan, nan, nan, nan

    low, high = np.nanpercentile(samples, STAT_PERCENTILES, axis=0)
    err_minus = np.abs(nominal - low)
    err_plus = np.abs(high - nominal)
    sigma = 0.5 * (err_minus + err_plus)
    return low, high, err_minus, err_plus, sigma


def summarize_parameter_toys(
    nominal: np.ndarray,
    stat_samples: np.ndarray,
    syst_samples: np.ndarray,
    names_csv: list[str],
) -> pd.DataFrame:
    nominal = np.asarray(nominal, dtype=float)
    stat_low, stat_high, stat_minus, stat_plus, stat_sigma = _safe_percentiles(stat_samples, nominal)
    syst_low, syst_high, syst_minus, syst_plus, syst_sigma = _safe_percentiles(syst_samples, nominal)
    total_sigma = np.sqrt(np.nan_to_num(stat_sigma, nan=0.0) ** 2 + np.nan_to_num(syst_sigma, nan=0.0) ** 2)

    return pd.DataFrame(
        {
            "parameter_name": names_csv,
            "value": nominal,
            "stat_p16": stat_low,
            "stat_p84": stat_high,
            "stat_minus": stat_minus,
            "stat_plus": stat_plus,
            "stat_sigma": stat_sigma,
            "syst_p16": syst_low,
            "syst_p84": syst_high,
            "syst_minus": syst_minus,
            "syst_plus": syst_plus,
            "syst_sigma": syst_sigma,
            "total_sigma_quadrature": total_sigma,
        }
    )


def _sci(x: float, sig: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return r"--"
    if x == 0:
        return r"\num{0}"
    return rf"\num{{{float(x):.{sig}g}}}"


def _asym(value_minus: float, value_plus: float, sig: int = 2) -> str:
    if not np.isfinite(value_minus) or not np.isfinite(value_plus):
        return r"--"
    return rf"$^{{+{value_plus:.{sig}g}}}_{{-{value_minus:.{sig}g}}}$"


def export_parameter_uncertainty_table(
    df: pd.DataFrame,
    names_tex: list[str],
    filename: str | Path,
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Parámetro & Valor & Estadístico & Sistemático \\",
        r"\midrule",
    ]

    for i, pname in enumerate(names_tex):
        row = df.iloc[i]
        lines.append(
            f"{pname} & {_sci(row['value'])} & "
            f"{_asym(row['stat_minus'], row['stat_plus'])} & "
            f"{_asym(row['syst_minus'], row['syst_plus'])} \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    text = "\n".join(lines)
    path = Path(filename)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return text


def save_toy_parameter_payload(
    output_npz: str | Path,
    nominal: np.ndarray,
    stat_samples: np.ndarray,
    syst_samples: np.ndarray,
    names_csv: list[str],
) -> None:
    ensure_dir(Path(output_npz).parent)
    np.savez_compressed(
        output_npz,
        nominal=np.asarray(nominal, dtype=float),
        stat=np.asarray(stat_samples, dtype=float),
        syst=np.asarray(syst_samples, dtype=float),
        names=np.asarray(names_csv, dtype=object),
    )


def load_toy_parameter_payload(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def build_statistical_toy_dict(
    exp_stat: Mapping[str, pd.DataFrame],
    bar_cols: list[str],
    err_cols: list[str],
    rng: np.random.Generator,
    *,
    clip_nonnegative: bool = True,
) -> dict[str, pd.DataFrame]:
    toys = {}
    for key, df in exp_stat.items():
        toy = df.copy(deep=True)
        cols = [c for c in bar_cols if c in toy.columns]
        errs = [f"Err {c}" for c in cols]
        y = toy[cols].to_numpy(dtype=float)
        e = toy[errs].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(e) & (e > 0)
        y_toy = np.where(mask, y + rng.normal(0.0, 1.0, size=y.shape) * e, y)
        if clip_nonnegative:
            y_toy = np.where(mask, np.maximum(y_toy, 0.0), y_toy)
        toy.loc[:, cols] = y_toy
        toys[key] = toy.fillna(0)
    return toys


def build_correlated_systematic_toy_dict(
    exp_central: Mapping[str, pd.DataFrame],
    exp_sys: Mapping[str, pd.DataFrame],
    bar_cols: list[str],
    err_cols: list[str],
    rng: np.random.Generator,
    *,
    group_map: Mapping[str, str] | None = None,
    clip_nonnegative: bool = True,
) -> dict[str, pd.DataFrame]:
    """One Gaussian nuisance per group, shared by all valid points in that group."""
    if group_map is None:
        group_map = {key: key for key in exp_central}

    unique_groups = sorted(set(group_map.values()))
    z = {group: rng.normal(0.0, 1.0) for group in unique_groups}

    toys = {}
    for key, df_c in exp_central.items():
        df_s = exp_sys[key]
        toy = df_c.copy(deep=True)
        cols = [c for c in bar_cols if c in toy.columns]
        errs = [f"Err {c}" for c in cols]
        y = df_c[cols].to_numpy(dtype=float)
        e_sys = df_s[errs].to_numpy(dtype=float)
        e_fit = df_c[errs].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(e_sys)
        y_toy = np.where(mask, y + z[group_map[key]] * e_sys, y)
        if clip_nonnegative:
            y_toy = np.where(mask, np.maximum(y_toy, 0.0), y_toy)
        toy.loc[:, cols] = y_toy
        # Keep the statistical errors as fit weights.  The systematic is the toy shift,
        # not an independent point-wise denominator.
        toy.loc[:, errs] = np.where(np.isfinite(e_fit), e_fit, np.nan)
        toys[key] = toy.fillna(0)
    return toys


def fit_toy_parameters(
    n_toys: int,
    toy_builder: Callable[[np.random.Generator], dict[str, pd.DataFrame]],
    fit_runner: Callable[[dict[str, pd.DataFrame], np.ndarray], object],
    nominal: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    params = []
    failures = 0
    for _ in range(n_toys):
        try:
            toy_data = toy_builder(rng)
            toy_fit = fit_runner(toy_data, nominal)
            x = np.asarray(toy_fit.x, dtype=float)
            if np.all(np.isfinite(x)):
                params.append(x)
            else:
                failures += 1
        except Exception:
            failures += 1
    if len(params) == 0:
        return np.empty((0, len(nominal)), dtype=float), failures
    return np.asarray(params, dtype=float), failures


def percentile_curve_band(curves: np.ndarray, nominal_curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[0] == 0:
        return np.asarray(nominal_curve, dtype=float).copy(), np.asarray(nominal_curve, dtype=float).copy()
    return tuple(np.nanpercentile(curves, STAT_PERCENTILES, axis=0))


def curves_from_parameters(params: np.ndarray, model: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[0] == 0:
        return np.empty((0, 0), dtype=float)
    curves = []
    for par in params:
        try:
            curves.append(np.asarray(model(par), dtype=float))
        except Exception:
            continue
    if len(curves) == 0:
        return np.empty((0, 0), dtype=float)
    return np.asarray(curves, dtype=float)


def save_band_csv(
    output_csv: str | Path,
    x_percent: np.ndarray,
    y_nominal: np.ndarray,
    stat_low: np.ndarray,
    stat_high: np.ndarray,
    syst_low: np.ndarray,
    syst_high: np.ndarray,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    df = pd.DataFrame(
        {
            "x_percent": np.asarray(x_percent, dtype=float),
            "y_nominal": np.asarray(y_nominal, dtype=float),
            "stat_low": np.asarray(stat_low, dtype=float),
            "stat_high": np.asarray(stat_high, dtype=float),
            "syst_low": np.asarray(syst_low, dtype=float),
            "syst_high": np.asarray(syst_high, dtype=float),
        }
    )
    if metadata:
        for key, value in metadata.items():
            df[str(key)] = value
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)


def export_parameter_products(
    data_dir: str | Path,
    tex_dir: str | Path,
    basename: str,
    names_csv: list[str],
    names_tex: list[str],
    nominal: np.ndarray,
    stat_params: np.ndarray,
    syst_params: np.ndarray,
    *,
    caption: str,
    label: str,
) -> pd.DataFrame:
    data_dir = ensure_dir(data_dir)
    tex_dir = ensure_dir(tex_dir)
    df = summarize_parameter_toys(nominal, stat_params, syst_params, names_csv)
    df.to_csv(data_dir / f"{basename}_parameter_uncertainties.csv", index=False)
    save_toy_parameter_payload(data_dir / f"{basename}_toy_parameters.npz", nominal, stat_params, syst_params, names_csv)
    export_parameter_uncertainty_table(
        df,
        names_tex,
        tex_dir / f"{basename}_param_stat_syst.tex",
        caption=caption,
        label=label,
    )
    return df
