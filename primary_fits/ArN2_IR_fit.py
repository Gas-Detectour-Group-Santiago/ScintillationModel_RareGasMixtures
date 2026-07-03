from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for folder in ("models", "primary_fits"):
    path = str(PROJECT_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

from auxiliares import DatasetSpec, FitConfig, Parameter, PlotSpec, PrimaryFitRunner, SystematicSource, ToySpec
from auxiliares.fit_io import error_label, pressure_label, stat_error_label, syst_error_label
from ArN2_infrarred import (
    W_ArN2,
    theory_yield_ArN2_Ir_696,
    theory_yield_ArN2_Ir_727,
    theory_yield_ArN2_Ir_750,
    theory_yield_ArN2_Ir_763,
    theory_yield_ArN2_Ir_772,
)


DATA_DIR = PROJECT_ROOT / "data"
IR_LINES = ("696", "727", "750", "763", "772")
TAUS = {"696": 28.3, "727": 28.3, "750": 21.7, "763": 29.4, "772": 28.3}

# ---------------------------------------------------------------------------
# Selección IR opcional, análoga a ArCF4
# ---------------------------------------------------------------------------
# Por defecto se deja el fit exactamente con el criterio anterior:
#   - presiones 1, 2, 3 bar;
#   - concentraciones de N2 estrictamente menores que 10 %.
#
# Para activar el mismo criterio legacy-floor usado en ArCF4:
#
#     ARN2_IR_SELECTION_MODE=legacy_floor python3 primary_fits/ArN2_IR_fit.py
#
# Ese modo:
#   1) mueve el punto de argón puro fN2=0 a x=0.001 % para escala log;
#   2) descarta 20 %, 50 % y 100 % N2;
#   3) calcula floor = min_p max(Y_20,Y_50,Y_100) con p=1,2,3 bar;
#   4) elimina, solo para 1,2,3 bar, cualquier punto con Y < floor.
#
# No se regeneran parámetros aquí: cambiar este archivo solo afecta a futuros
# fits si se ejecutan de nuevo.
IR_SELECTION_MODE = os.environ.get("ARN2_IR_SELECTION_MODE", "none").strip().lower()
IR_FIT_PRESSURES = (1.0, 2.0, 3.0, 4, 5)
IR_DEFAULT_MAX_CONCENTRATION_PERCENT = 50.0
IR_DISCARDED_CONCENTRATIONS_PERCENT = (10.0, 20.0, 50.0, 100.0)
IR_PURE_ARGON_DISPLAY_PERCENT = 0.0
IR_PLOT_MAX_CONCENTRATION_PERCENT = 100.0
IR_FIRST_POINT_ANCHOR_ENABLED = os.environ.get(
    "ARN2_IR_FIRST_POINT_ANCHOR_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
IR_FIRST_POINT_ANCHOR_WEIGHT = float(os.environ.get("ARN2_IR_FIRST_POINT_ANCHOR_WEIGHT", "25.0"))
if not IR_FIRST_POINT_ANCHOR_ENABLED:
    IR_FIRST_POINT_ANCHOR_WEIGHT = 1.0


def _arn2_ir_csv_path(line: str) -> Path:
    """Return the IR experimental CSV path, tolerating both project layouts."""

    candidates = (
        DATA_DIR / "Experimental" / "ArN2" / "csv" / f"{line}.csv",
        DATA_DIR / "Experimental" / "ArN2" / f"{line}.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _numeric_array(values) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def move_pure_argon_to_low_n2(df: pd.DataFrame) -> pd.DataFrame:
    """Move fN2=0 to 0.001 % for log-x plotting and ArCF4-like IR handling."""

    out = df.copy()
    if "fN2" not in out.columns:
        return out
    x = _numeric_array(out["fN2"])
    mask = np.isclose(x, 0.0, rtol=0.0, atol=1e-12)
    out.loc[mask, "fN2"] = IR_PURE_ARGON_DISPLAY_PERCENT
    return out.sort_values("fN2").reset_index(drop=True)


def _discarded_concentration_mask(x_percent: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(x_percent, dtype=bool)
    for c in IR_DISCARDED_CONCENTRATIONS_PERCENT:
        mask |= np.isclose(x_percent, c, rtol=0.0, atol=1e-10)
    return mask


def _legacy_floor_from_discarded_points(df: pd.DataFrame) -> float | None:
    """Compute min_p max(Y_20,Y_50,Y_100) for p=1,2,3 bar."""

    if "fN2" not in df.columns:
        return None

    x_percent = _numeric_array(df["fN2"])
    discarded = _discarded_concentration_mask(x_percent)
    maxima: list[float] = []

    for pressure in IR_FIT_PRESSURES:
        y_col = pressure_label(pressure)
        if y_col not in df.columns:
            continue
        y = _numeric_array(df[y_col])
        vals = y[discarded & np.isfinite(y) & (y > 0.0)]
        if vals.size:
            maxima.append(float(np.nanmax(vals)))

    if not maxima:
        return None
    return float(np.nanmin(maxima))


def make_legacy_ir_selector(selection_mode: str = IR_SELECTION_MODE):
    """Mask ArN2 IR cells using the same legacy-floor criterion as ArCF4."""

    if selection_mode not in {"legacy_floor", "none"}:
        raise ValueError("ARN2_IR_SELECTION_MODE debe ser 'legacy_floor' o 'none'.")

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if selection_mode == "none":
            return out
        if "fN2" not in out.columns:
            return out

        x_percent = _numeric_array(out["fN2"])
        discarded = _discarded_concentration_mask(x_percent)
        floor = _legacy_floor_from_discarded_points(out)

        for pressure in IR_FIT_PRESSURES:
            y_col = pressure_label(pressure)
            err_cols = [
                error_label(pressure),
                stat_error_label(pressure),
                syst_error_label(pressure),
            ]
            cols_to_mask = [c for c in [y_col, *err_cols] if c in out.columns]
            if y_col not in out.columns or not cols_to_mask:
                continue

            y = _numeric_array(out[y_col])
            bad = discarded.copy()
            bad |= ~np.isfinite(y) | (y <= 0.0)
            if floor is not None and np.isfinite(floor):
                bad |= y < floor

            err_ref = None
            for candidate in (error_label(pressure), stat_error_label(pressure)):
                if candidate in out.columns:
                    err_ref = _numeric_array(out[candidate])
                    break
            if err_ref is not None:
                bad |= ~np.isfinite(err_ref) | (err_ref <= 0.0)

            if np.any(bad):
                out.loc[bad, cols_to_mask] = np.nan

        return out

    return preprocess


def ir_parameters():
    params = []
    for line in IR_LINES:
        tau = TAUS[line]
        display = "764" if line == "763" else line
        params.extend(
            [
                Parameter(
                    f"PAr_star_{display}",
                    rf"$\mathcal{{W}}_{{\mathrm{{Ar}}^{{**}},{display}\,\mathrm{{nm}}}}$",
                    0.0159,
                    0.0,
                    0.02,
                ),
                Parameter(
                    f"tau_N2_{display}",
                    rf"$\tau_{{\mathrm{{Ar}}^{{**}},{display}\,\mathrm{{nm}}}}$",
                    tau,
                    tau * 0.999999999999999,
                    tau * 1.000000000000001,
                    fixed=True,
                    fixed_value=tau,
                    fixed_error=0.1,
                ),
                Parameter(
                    f"K_Ar_Q_Ar_{display}",
                    rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{Ar}}),{display}\,\mathrm{{nm}}}}$",
                    1.0,
                    0.0,
                    1000.0,
                ),
                Parameter(
                    f"K_Ar_Q_N2_{display}",
                    rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{N}}_2),{display}\,\mathrm{{nm}}}}$",
                    1.0,
                    0.0,
                    1000.0,
                ),
            ]
        )
    return params


EQUATIONS = {
    "696": theory_yield_ArN2_Ir_696,
    "727": theory_yield_ArN2_Ir_727,
    "750": theory_yield_ArN2_Ir_750,
    "763": theory_yield_ArN2_Ir_763,
    "772": theory_yield_ArN2_Ir_772,
}


def build_datasets(selection_mode: str = IR_SELECTION_MODE) -> list[DatasetSpec]:
    legacy = selection_mode == "legacy_floor"
    return [
        DatasetSpec(
            key=line,
            csv_path=_arn2_ir_csv_path(line),
            x_col="fN2",
            pressures=IR_FIT_PRESSURES,
            output_concentration_name="fN2",
            w_function=W_ArN2,
            max_concentration_percent=None if legacy else IR_DEFAULT_MAX_CONCENTRATION_PERCENT,
            preprocess_before_w=move_pure_argon_to_low_n2 if legacy else None,
            preprocess=make_legacy_ir_selector(selection_mode) if legacy else None,
        )
        for line in IR_LINES
    ]


def build_plots(selection_mode: str = IR_SELECTION_MODE, *, output_subdir: str = "ArN2_IR") -> list[PlotSpec]:
    legacy = selection_mode == "legacy_floor"
    grid = np.logspace(-5, 0, 1000) if legacy else np.logspace(-6, np.log10(IR_DEFAULT_MAX_CONCENTRATION_PERCENT / 100.0), 1000)
    xlim = (IR_PURE_ARGON_DISPLAY_PERCENT, IR_PLOT_MAX_CONCENTRATION_PERCENT * 1.1) if legacy else (1e-3, IR_DEFAULT_MAX_CONCENTRATION_PERCENT * 1.1)
    return [
        PlotSpec(
            name=f"ArN2_IR_{line}",
            dataset_key=line,
            theory_key=line,
            pressures=IR_FIT_PRESSURES,
            concentration_grid=grid,
            title=rf"Ar--N$_2$ primary IR fit, {line} nm",
            xlabel=r"N$_2$ concentration [$\%$]",
            ylabel=r"Yield [arb. units]",
            x_col="fN2",
            min_positive_x=IR_PURE_ARGON_DISPLAY_PERCENT,
            xlim=xlim,
            #ylim=(1e-5, 0.09),
            output=PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / output_subdir / f"ArN2_global_{line}.pdf",
            legend_kwargs={"ncol": 2, "loc": "upper right"},
        )
        for line in IR_LINES
    ]


def build_config(
    selection_mode: str = IR_SELECTION_MODE,
    *,
    name: str = "ArN2_IR_primary",
    output_subdir: str = "ArN2_IR",
) -> FitConfig:
    return FitConfig(
        name=name,
        model_name="ArN2_infrarred",
        degrad_csv=DATA_DIR / "Primary_DegradData" / "ArN2_IR.csv",
        datasets=build_datasets(selection_mode),
        equations=EQUATIONS,
        parameters=ir_parameters(),
        plots=build_plots(selection_mode, output_subdir=output_subdir),
        is_infrared=True,
        first_point_anchor_weight=IR_FIRST_POINT_ANCHOR_WEIGHT,
        toy_spec=ToySpec(
            n_stat=5,
            n_syst=5,
            seed=44001,
            n_jobs=-1,
            syst_sources=tuple(SystematicSource(f"line_{line}_calibration", mode="by_dataset", datasets=(line,)) for line in IR_LINES),
        ),
        table_caption=r"Parámetros del ajuste primario IR en Ar--N$_2$.",
        table_label="tab:ArN2_IR_primary_stat_syst",
    )


CONFIG = build_config(IR_SELECTION_MODE)


if __name__ == "__main__":
    PrimaryFitRunner(CONFIG, project_root=PROJECT_ROOT).run_all()
