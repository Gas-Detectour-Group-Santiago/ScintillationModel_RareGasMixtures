#!/usr/bin/env python3
"""Joint IR fit of Ar--CF4 and Ar--N2.

This is deliberately an *additional* fit.  It does not overwrite the legacy
ArCF4_IR_primary or ArN2_IR_primary products.

Shared by both mixtures for every Ar line:
  - PAr_star (optical weight / low-pressure amplitude)
  - K_Ar_Q_Ar (Ar self-quenching)

Independent:
  - K_Ar_Q_CF4
  - K_Ar_Q_N2

The residuals are evaluated at the exact experimental concentrations.  No
array-length trimming and no first-point anchor are used.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for folder in ("models", "primary_fits"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

from ArCF4_infrarred import W_ArCF4
from ArN2_infrarred import W_ArN2
from ArJoint_infrarred import IR_LINES, TAUS_NS, interpolate_population, theory_yield_joint
from auxiliares.fit_io import DatasetSpec, load_dataset_triplet, pressure_label, error_label, stat_error_label, syst_error_label

DATA_DIR = PROJECT_ROOT / "data"
FIT_NAME = "ArJoint_IR_primary"
PRESSURES = (1.0, 2.0, 3.0)
MAX_CONCENTRATION_PERCENT = float(os.environ.get("JOINT_IR_MAX_CONCENTRATION_PERCENT", "20.0"))
N_TOYS = int(os.environ.get("JOINT_IR_N_TOYS", "300"))
SEED = int(os.environ.get("JOINT_IR_SEED", "55001"))


@dataclass(frozen=True)
class ObservationBlock:
    mixture: str
    line: str
    pressure: float
    concentration: np.ndarray
    value: np.ndarray
    stat: np.ndarray
    syst: np.ndarray
    population: np.ndarray


PARAMETER_NAMES: list[str] = []
PARAMETER_TEX: list[str] = []
FIXED_MASK: list[bool] = []
for line in IR_LINES:
    display = "764" if line == "763" else line
    PARAMETER_NAMES.extend(
        [
            f"PAr_star_{display}",
            f"tau_joint_{display}",
            f"K_Ar_Q_Ar_{display}",
            f"K_Ar_Q_CF4_{display}",
            f"K_Ar_Q_N2_{display}",
        ]
    )
    PARAMETER_TEX.extend(
        [
            rf"$\mathcal{{W}}_{{\mathrm{{Ar}}^{{**}},{display}\,\mathrm{{nm}}}}$",
            rf"$\tau_{{\mathrm{{Ar}}^{{**}},{display}\,\mathrm{{nm}}}}$",
            rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{Ar}}),{display}\,\mathrm{{nm}}}}$",
            rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{CF}}_4),{display}\,\mathrm{{nm}}}}$",
            rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{N}}_2),{display}\,\mathrm{{nm}}}}$",
        ]
    )
    FIXED_MASK.extend([False, True, False, False, False])

FREE_INDICES = np.asarray([i for i, fixed in enumerate(FIXED_MASK) if not fixed], dtype=int)


def _experimental_path(mixture: str, line: str) -> Path:
    return DATA_DIR / "Experimental" / mixture / "csv" / f"{line}.csv"


def _dataset_spec(mixture: str, line: str) -> DatasetSpec:
    x_col = "fCF4" if mixture == "ArCF4" else "fN2"
    w_fun = W_ArCF4 if mixture == "ArCF4" else W_ArN2
    # _apply_concentration_window uses '<', so a tiny epsilon includes the
    # requested endpoint (20% by default).
    return DatasetSpec(
        key=f"{mixture}_{line}",
        csv_path=_experimental_path(mixture, line),
        x_col=x_col,
        pressures=PRESSURES,
        output_concentration_name=x_col,
        w_function=w_fun,
        max_concentration_percent=MAX_CONCENTRATION_PERCENT + 1e-9,
    )


def load_observations() -> tuple[list[ObservationBlock], dict[str, pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    degrad = {
        "ArCF4": pd.read_csv(DATA_DIR / "Primary_DegradData" / "ArCF4_IR.csv"),
        "ArN2": pd.read_csv(DATA_DIR / "Primary_DegradData" / "ArN2_IR.csv"),
    }
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    blocks: list[ObservationBlock] = []

    for mixture in ("ArCF4", "ArN2"):
        x_col = "fCF4" if mixture == "ArCF4" else "fN2"
        for line in IR_LINES:
            triplet = load_dataset_triplet(PROJECT_ROOT, _dataset_spec(mixture, line))
            df = triplet["all"].copy()
            frames[(mixture, line)] = df
            f = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float) * 0.01

            for pressure in PRESSURES:
                y = pd.to_numeric(df[pressure_label(pressure)], errors="coerce").to_numpy(dtype=float)
                stat = np.abs(pd.to_numeric(df[stat_error_label(pressure)], errors="coerce").to_numpy(dtype=float))
                syst = np.abs(pd.to_numeric(df[syst_error_label(pressure)], errors="coerce").to_numpy(dtype=float))
                valid = np.isfinite(f) & np.isfinite(y) & np.isfinite(stat) & np.isfinite(syst) & (y > 0.0)
                valid &= np.sqrt(stat**2 + syst**2) > 0.0
                if np.any(valid):
                    blocks.append(
                        ObservationBlock(
                            mixture=mixture,
                            line=line,
                            pressure=pressure,
                            concentration=f[valid],
                            value=y[valid],
                            stat=stat[valid],
                            syst=syst[valid],
                            population=np.asarray(interpolate_population(degrad[mixture], line, f[valid]), dtype=float),
                        )
                    )
    return blocks, degrad, frames


def _read_legacy_parameter_table(mixture: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Parameters" / f"{mixture}_IR_primary.csv").set_index("name")


@lru_cache(maxsize=1)
def _initial_physical_tuple() -> tuple[float, ...]:
    cf4 = _read_legacy_parameter_table("ArCF4")
    n2 = _read_legacy_parameter_table("ArN2")
    values: list[float] = []

    def get(table: pd.DataFrame, name: str, fallback: float) -> float:
        if name not in table.index:
            return fallback
        value = float(table.loc[name, "value"])
        return value if np.isfinite(value) and value > 0.0 else fallback

    for line in IR_LINES:
        display = "764" if line == "763" else line
        p1 = get(cf4, f"PAr_star_{display}", 1e-3)
        p2 = get(n2, f"PAr_star_{display}", 1e-3)
        kar1 = get(cf4, f"K_Ar_Q_Ar_{display}", 0.1)
        kar2 = get(n2, f"K_Ar_Q_Ar_{display}", 0.1)
        kcf4 = get(cf4, f"K_Ar_Q_CF4_{display}", 1.0)
        kn2 = get(n2, f"K_Ar_Q_N2_{display}", 1.0)
        values.extend(
            [
                float(np.sqrt(p1 * p2)),
                TAUS_NS[line],
                float(np.sqrt(kar1 * kar2)),
                max(kcf4, 1e-5),
                max(kn2, 1e-5),
            ]
        )
    return tuple(float(v) for v in values)


def initial_physical_vector() -> np.ndarray:
    return np.asarray(_initial_physical_tuple(), dtype=float).copy()


def physical_to_free_log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(x, dtype=float)[FREE_INDICES], 1e-12, None))


def free_log10_to_physical(z: np.ndarray) -> np.ndarray:
    x = initial_physical_vector()
    x[FREE_INDICES] = 10.0 ** np.asarray(z, dtype=float)
    for i, line in enumerate(IR_LINES):
        x[5 * i + 1] = TAUS_NS[line]
    return x


def residual_vector(
    z: np.ndarray,
    blocks: list[ObservationBlock],
    degrad: dict[str, pd.DataFrame],
    *,
    toy_values: list[np.ndarray] | None = None,
) -> np.ndarray:
    x = free_log10_to_physical(z)
    chunks: list[np.ndarray] = []
    for ib, block in enumerate(blocks):
        y = block.value if toy_values is None else toy_values[ib]
        sigma = np.sqrt(block.stat**2 + block.syst**2)
        offset = 5 * IR_LINES.index(block.line)
        P = x[offset]
        tau = x[offset + 1]
        k_ar = x[offset + 2]
        k_mol = x[offset + 3] if block.mixture == "ArCF4" else x[offset + 4]
        radiative = 1.0 / tau
        survival = P * radiative / (
            radiative
            + block.pressure * (1.0 - block.concentration) * k_ar
            + block.pressure * block.concentration * k_mol
        )
        energy_kev = 15.0 if block.mixture == "ArCF4" else 12.0
        y_th = survival * block.population / energy_kev
        valid = np.isfinite(y) & np.isfinite(y_th) & np.isfinite(sigma) & (sigma > 0.0)
        if np.any(valid):
            chunks.append((y[valid] - y_th[valid]) / sigma[valid])
    if not chunks:
        raise RuntimeError("Joint IR fit has no valid residuals.")
    return np.concatenate(chunks)


def fit_once(
    blocks: list[ObservationBlock],
    degrad: dict[str, pd.DataFrame],
    *,
    z0: np.ndarray,
    toy_values: list[np.ndarray] | None = None,
    fast: bool = False,
):
    lower: list[float] = []
    upper: list[float] = []
    w_max = float(
        pd.read_csv(DATA_DIR / "Parameters" / "ArCF4_primary.csv")
        .query("name == 'Nnorm'")["value"]
        .iloc[0]
    )
    for line in IR_LINES:
        lower.extend([np.log10(1e-9), np.log10(1e-7), np.log10(1e-7), np.log10(1e-7)])
        upper.extend([np.log10(w_max), np.log10(1000.0), np.log10(1000.0), np.log10(1000.0)])

    return least_squares(
        lambda z: residual_vector(z, blocks, degrad, toy_values=toy_values),
        np.asarray(z0, dtype=float),
        bounds=(np.asarray(lower), np.asarray(upper)),
        method="trf",
        max_nfev=700 if fast else 5000,
        xtol=1e-7 if fast else 1e-11,
        ftol=1e-7 if fast else 1e-11,
        gtol=1e-7 if fast else 1e-11,
    )


def covariance_physical(result, central_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = np.asarray(result.jac, dtype=float)
    m, n = j.shape
    try:
        _, s, vt = np.linalg.svd(j, full_matrices=False)
        threshold = np.finfo(float).eps * max(j.shape) * s[0]
        keep = s > threshold
        inv = (vt[keep].T / (s[keep] ** 2)) @ vt[keep]
        dof = m - n
        cov_z = inv * (2.0 * result.cost / dof) if dof > 0 else np.full((n, n), np.nan)
    except (np.linalg.LinAlgError, IndexError):
        cov_z = np.full((n, n), np.nan)

    cov_x = np.full((len(central_x), len(central_x)), np.nan)
    transform = np.diag(np.log(10.0) * central_x[FREE_INDICES])
    cov_free_x = transform @ cov_z @ transform
    cov_x[np.ix_(FREE_INDICES, FREE_INDICES)] = cov_free_x
    for i, fixed in enumerate(FIXED_MASK):
        if fixed:
            cov_x[i, i] = 0.1**2
    err = np.sqrt(np.clip(np.diag(cov_x), 0.0, None))
    return cov_x, err


def make_toy_values(blocks: list[ObservationBlock], rng: np.random.Generator, mode: str) -> list[np.ndarray]:
    if mode == "stat":
        return [block.value + rng.normal(size=block.value.size) * block.stat for block in blocks]
    if mode == "syst":
        # One correlated calibration displacement per mixture and spectral line,
        # shared by all pressures and concentrations in that family.
        shifts = {(mix, line): rng.normal() for mix in ("ArCF4", "ArN2") for line in IR_LINES}
        return [block.value + shifts[(block.mixture, block.line)] * block.syst for block in blocks]
    raise ValueError(mode)


def run_toys(
    blocks: list[ObservationBlock],
    degrad: dict[str, pd.DataFrame],
    central_z: np.ndarray,
    *,
    n_toys: int,
    seed: int,
    mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for itoy in range(n_toys):
        toy = make_toy_values(blocks, rng, mode)
        result = fit_once(blocks, degrad, z0=central_z, toy_values=toy, fast=True)
        if result.success and np.all(np.isfinite(result.x)):
            rows.append(free_log10_to_physical(result.x))
        if (itoy + 1) % max(1, n_toys // 10) == 0:
            print(f"[ArJoint IR] {mode} toys: {itoy + 1}/{n_toys}")
    if not rows:
        return np.empty((0, len(PARAMETER_NAMES)))
    return np.vstack(rows)


def asymmetric_spread(toys: np.ndarray, central: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if toys.size == 0:
        nan = np.full_like(central, np.nan, dtype=float)
        return nan, nan
    q16, q84 = np.nanpercentile(toys, [16.0, 84.0], axis=0)
    return np.maximum(central - q16, 0.0), np.maximum(q84 - central, 0.0)


def export_parameter_table(summary: pd.DataFrame) -> None:
    out = DATA_DIR / "Tables" / f"{FIT_NAME}_param_stat_syst.tex"
    out.parent.mkdir(parents=True, exist_ok=True)

    def num(v: float) -> str:
        return r"--" if not np.isfinite(v) else rf"\num{{{v:.3g}}}"

    def asym(m: float, p: float) -> str:
        if not np.isfinite(m) or not np.isfinite(p) or (m == 0 and p == 0):
            return r"--"
        return rf"$^{{+{num(p)}}}_{{-{num(m)}}}$"

    header = [
        r"\toprule",
        r"Parámetro & Valor & Stat. & Syst. & Total \\",
        r"\midrule",
    ]
    lines = [
        r"\begin{longtable}{lcccc}",
        r"\caption{Parámetros del ajuste conjunto de las líneas IR de Ar en Ar--CF$_4$ y Ar--N$_2$. Los pesos ópticos y el quenching por Ar son comunes; los coeficientes de desactivación por CF$_4$ y N$_2$ se ajustan de forma independiente.}",
        r"\label{tab:ArJoint_IR_primary_stat_syst}\\",
        *header,
        r"\endfirsthead",
        *header,
        r"\endhead",
    ]
    for irow, (_, row) in enumerate(summary.iterrows()):
        lines.append(
            f"{row['tex_name']} & {num(row['value'])} & "
            f"{asym(row['stat_minus'], row['stat_plus'])} & "
            f"{asym(row['syst_minus'], row['syst_plus'])} & "
            f"{asym(row['total_minus'], row['total_plus'])} \\\\"
        )
        if (irow + 1) % 5 == 0 and (irow + 1) < len(summary):
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def plot_fit_diagnostics(
    central_x: np.ndarray,
    degrad: dict[str, pd.DataFrame],
    frames: dict[tuple[str, str], pd.DataFrame],
) -> None:
    output_dir = PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / "ArJoint_IR"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")

    for mixture in ("ArCF4", "ArN2"):
        x_col = "fCF4" if mixture == "ArCF4" else "fN2"
        for line in IR_LINES:
            fig, ax = plt.subplots(figsize=(6.4, 4.5))
            df = frames[(mixture, line)]
            grid_percent = np.geomspace(1e-3, MAX_CONCENTRATION_PERCENT, 500)
            for ip, pressure in enumerate(PRESSURES):
                color = cmap((ip + 0.7) / (len(PRESSURES) + 0.4))
                y = pd.to_numeric(df[pressure_label(pressure)], errors="coerce").to_numpy(dtype=float)
                e = pd.to_numeric(df[error_label(pressure)], errors="coerce").to_numpy(dtype=float)
                x_percent = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(x_percent) & np.isfinite(y) & np.isfinite(e) & (y > 0) & (e > 0)
                x_plot = x_percent[valid].copy()
                x_plot[np.isclose(x_plot, 0.0)] = 1e-3
                ax.errorbar(x_plot, y[valid], yerr=e[valid], fmt="o", ms=3.5, capsize=2, color=color, alpha=0.8)
                y_th = theory_yield_joint(
                    central_x,
                    degrad[mixture],
                    grid_percent * 0.01,
                    pressure,
                    mixture=mixture,
                    line=line,
                )
                ax.plot(grid_percent, y_th, lw=1.7, color=color, label=f"{pressure:g} bar")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-3, MAX_CONCENTRATION_PERCENT * 1.08)
            ax.set_xlabel((r"CF$_4$" if mixture == "ArCF4" else r"N$_2$") + " concentration [%]")
            ax.set_ylabel("Yield [arb. units]")
            ax.set_title(rf"Joint IR fit: {mixture}, {line} nm")
            ax.legend(ncol=3, fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"ArJoint_{mixture}_{line}.pdf", bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    blocks, degrad, frames = load_observations()
    x0 = initial_physical_vector()
    z0 = physical_to_free_log10(x0)
    central_result = fit_once(blocks, degrad, z0=z0)
    if not central_result.success:
        raise RuntimeError(f"Joint IR central fit failed: {central_result.message}")

    central_x = free_log10_to_physical(central_result.x)
    covariance, fit_err = covariance_physical(central_result, central_x)
    stat_toys = run_toys(blocks, degrad, central_result.x, n_toys=N_TOYS, seed=SEED, mode="stat")
    syst_toys = run_toys(blocks, degrad, central_result.x, n_toys=N_TOYS, seed=SEED + 1, mode="syst")
    stat_minus, stat_plus = asymmetric_spread(stat_toys, central_x)
    syst_minus, syst_plus = asymmetric_spread(syst_toys, central_x)
    total_minus = np.sqrt(stat_minus**2 + syst_minus**2)
    total_plus = np.sqrt(stat_plus**2 + syst_plus**2)

    summary = pd.DataFrame(
        {
            "name": PARAMETER_NAMES,
            "tex_name": PARAMETER_TEX,
            "value": central_x,
            "fit_uncertainty": fit_err,
            "stat_minus": stat_minus,
            "stat_plus": stat_plus,
            "syst_minus": syst_minus,
            "syst_plus": syst_plus,
            "total_minus": total_minus,
            "total_plus": total_plus,
            "fixed": FIXED_MASK,
        }
    )

    fit_dir = DATA_DIR / "FitResults"
    param_dir = DATA_DIR / "Parameters"
    fit_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(param_dir / f"{FIT_NAME}.csv", index=False)
    summary.to_csv(fit_dir / f"{FIT_NAME}_central.csv", index=False)
    pd.DataFrame(stat_toys, columns=PARAMETER_NAMES).to_csv(fit_dir / f"{FIT_NAME}_toys_stat.csv", index=False)
    pd.DataFrame(syst_toys, columns=PARAMETER_NAMES).to_csv(fit_dir / f"{FIT_NAME}_toys_syst.csv", index=False)
    pd.DataFrame(covariance, index=PARAMETER_NAMES, columns=PARAMETER_NAMES).to_csv(
        fit_dir / f"{FIT_NAME}_covariance.csv"
    )
    denom = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    correlation = np.divide(covariance, denom, out=np.full_like(covariance, np.nan), where=denom > 0)
    pd.DataFrame(correlation, index=PARAMETER_NAMES, columns=PARAMETER_NAMES).to_csv(
        fit_dir / f"{FIT_NAME}_correlation.csv"
    )

    n_residuals = central_result.fun.size
    n_free = central_result.x.size
    metadata = {
        "fit_name": FIT_NAME,
        "success": bool(central_result.success),
        "message": str(central_result.message),
        "chi2": float(2.0 * central_result.cost),
        "ndf": int(n_residuals - n_free),
        "chi2_ndf": float(2.0 * central_result.cost / (n_residuals - n_free)),
        "n_residuals": int(n_residuals),
        "n_free_parameters": int(n_free),
        "n_stat_toys_valid": int(len(stat_toys)),
        "n_syst_toys_valid": int(len(syst_toys)),
        "max_concentration_percent": MAX_CONCENTRATION_PERCENT,
        "pressures_bar": list(PRESSURES),
        "shared_parameters": ["PAr_star", "K_Ar_Q_Ar"],
        "independent_parameters": ["K_Ar_Q_CF4", "K_Ar_Q_N2"],
        "exact_experimental_concentrations": True,
        "first_point_anchor_weight": 1.0,
    }
    (fit_dir / f"{FIT_NAME}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    export_parameter_table(summary)
    plot_fit_diagnostics(central_x, degrad, frames)

    print(f"[ArJoint IR] chi2/ndf = {metadata['chi2_ndf']:.3f} ({metadata['chi2']:.1f}/{metadata['ndf']})")
    print(f"[ArJoint IR] parameters: {param_dir / f'{FIT_NAME}.csv'}")


if __name__ == "__main__":
    main()
