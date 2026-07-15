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
from ArCF4_infrarred import (
    W_ArCF4,
    theory_yield_ArCF4_Ir_696,
    theory_yield_ArCF4_Ir_727,
    theory_yield_ArCF4_Ir_750,
    theory_yield_ArCF4_Ir_763,
    theory_yield_ArCF4_Ir_772,
)


DATA_DIR = PROJECT_ROOT / "data"
IR_LINES = ("696", "727", "750", "763", "772")
TAUS = {"696": 28.3, "727": 28.3, "750": 21.7, "763": 29.4, "772": 28.3}

# ---------------------------------------------------------------------------
# Selección IR pedida
# ---------------------------------------------------------------------------
# 1) El punto de argón puro se representa en x = 0.001 % CF4.  El CSV fuente
#    no se toca: esta sustitución se aplica al DataFrame que entra al ajuste.
# 2) Para cada línea IR se descartan 20 %, 50 % y 100 % CF4.
# 3) Con esos puntos descartados se calcula, para 1, 2 y 3 bar:
#       max_descartado(p) = max(Y_20, Y_50, Y_100)
#    y se toma el menor de esos máximos:
#       floor = min_p max_descartado(p)
# 4) Solo para 1, 2 y 3 bar, se elimina cualquier punto con Y < floor.
#
# El modo por defecto es exactamente ese criterio.  Dejo el selector como
# variable de entorno para poder desactivarlo rápidamente si quieres comparar:
#
#     ARCF4_IR_SELECTION_MODE=none python3 primary_fits/ArCF4_IR_fit.py
#     ARCF4_IR_SELECTION_MODE=legacy_floor python3 primary_fits/ArCF4_IR_fit.py
IR_SELECTION_MODE = os.environ.get("ARCF4_IR_SELECTION_MODE", "legacy_floor").strip().lower()
IR_FIT_PRESSURES = (1.0, 2.0, 3.0)
IR_DISCARDED_CONCENTRATIONS_PERCENT = (50.0, 100.0) # (20.0, 50.0, 100.0)
IR_PURE_ARGON_DISPLAY_PERCENT = 0.001
IR_PLOT_MAX_CONCENTRATION_PERCENT = 100.0
IR_FIRST_POINT_ANCHOR_ENABLED = os.environ.get(
    "ARCF4_IR_FIRST_POINT_ANCHOR_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
IR_FIRST_POINT_ANCHOR_WEIGHT = float(os.environ.get("ARCF4_IR_FIRST_POINT_ANCHOR_WEIGHT", "25.0"))
if not IR_FIRST_POINT_ANCHOR_ENABLED:
    IR_FIRST_POINT_ANCHOR_WEIGHT = 1.0


def _arcf4_ir_csv_path(line: str) -> Path:
    """Return the IR experimental CSV path, tolerating both project layouts."""

    candidates = (
        DATA_DIR / "Experimental" / "ArCF4" / "csv" / f"{line}.csv",
        DATA_DIR / "Experimental" / "ArCF4" / f"{line}.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _numeric_array(values) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def move_pure_argon_to_low_cf4(df: pd.DataFrame) -> pd.DataFrame:
    """Move fCF4=0 to 0.001 % for log-x plotting and old IR alignment."""

    out = df.copy()
    if "fCF4" not in out.columns:
        return out
    x = _numeric_array(out["fCF4"])
    mask = np.isclose(x, 0.0, rtol=0.0, atol=1e-12)
    out.loc[mask, "fCF4"] = IR_PURE_ARGON_DISPLAY_PERCENT
    return out.sort_values("fCF4").reset_index(drop=True)


def _discarded_concentration_mask(x_percent: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(x_percent, dtype=bool)
    for c in IR_DISCARDED_CONCENTRATIONS_PERCENT:
        mask |= np.isclose(x_percent, c, rtol=0.0, atol=1e-10)
    return mask


def _legacy_floor_from_discarded_points(df: pd.DataFrame) -> float | None:
    """Compute min_p max(Y_20,Y_50,Y_100) for p=1,2,3 bar."""

    if "fCF4" not in df.columns:
        return None

    x_percent = _numeric_array(df["fCF4"])
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
    """Mask bad IR cells according to the requested legacy-floor criterion."""

    if selection_mode not in {"legacy_floor", "none"}:
        raise ValueError("ARCF4_IR_SELECTION_MODE debe ser 'legacy_floor' o 'none'.")

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if selection_mode == "none":
            return out
        if "fCF4" not in out.columns:
            return out

        x_percent = _numeric_array(out["fCF4"])
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

            # Si hay columna de error total/estadístico, evita que entren puntos
            # sin incertidumbre.  No se usa como criterio físico, solo sanitario.
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



def cf4_primary_norm_upper_bound() -> float:
    """Use the fitted Ar--CF4 Nnorm as the physical ceiling for every IR W."""

    candidates = (
        DATA_DIR / "FitResults" / "ArCF4_primary_central.csv",
        DATA_DIR / "Parameters" / "ArCF4_primary.csv",
    )
    for path in candidates:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "name" not in table.columns:
            continue
        rows = table.loc[table["name"].astype(str) == "Nnorm"]
        if rows.empty:
            continue
        for column in ("value",):
            if column in rows.columns:
                value = float(rows.iloc[0][column])
                if np.isfinite(value) and value > 0.0:
                    return value
    raise FileNotFoundError(
        "No se pudo leer Nnorm del ajuste ArCF4_primary; ejecuta primero el fit primario Ar--CF4."
    )

def ir_parameters():
    w_max = cf4_primary_norm_upper_bound()
    params = []
    for line in IR_LINES:
        tau = TAUS[line]
        display = "764" if line == "763" else line
        params.extend(
            [
                Parameter(
                    f"PAr_star_{display}",
                    rf"$\mathcal{{W}}_{{\mathrm{{Ar}}^{{**}},{display}\,\mathrm{{nm}}}}$",
                    min(0.02, 0.8 * w_max),
                    0.0,
                    w_max,
                ),
                Parameter(
                    f"tau_CF4_{display}",
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
                    f"K_Ar_Q_CF4_{display}",
                    rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{CF}}_4),{display}\,\mathrm{{nm}}}}$",
                    1.0,
                    0.0,
                    1000.0,
                ),
            ]
        )
    return params


EQUATIONS = {
    "696": theory_yield_ArCF4_Ir_696,
    "727": theory_yield_ArCF4_Ir_727,
    "750": theory_yield_ArCF4_Ir_750,
    "763": theory_yield_ArCF4_Ir_763,
    "772": theory_yield_ArCF4_Ir_772,
}


def build_datasets(selection_mode: str = IR_SELECTION_MODE) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key=line,
            csv_path=_arcf4_ir_csv_path(line),
            x_col="fCF4",
            pressures=IR_FIT_PRESSURES,
            output_concentration_name="fCF4",
            w_function=W_ArCF4,
            max_concentration_percent=None,
            preprocess_before_w=move_pure_argon_to_low_cf4,
            preprocess=make_legacy_ir_selector(selection_mode),
        )
        for line in IR_LINES
    ]


def build_plots(selection_mode: str = IR_SELECTION_MODE, *, output_subdir: str = "ArCF4_IR") -> list[PlotSpec]:
    return [
        PlotSpec(
            name=f"ArCF4_IR_{line}",
            dataset_key=line,
            theory_key=line,
            pressures=IR_FIT_PRESSURES,
            concentration_grid=np.logspace(-5, 0, 1000),
            title=rf"Ar--CF$_4$ primary IR fit, {line} nm",
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=r"Yield [arb. units]",
            x_col="fCF4",
            min_positive_x=IR_PURE_ARGON_DISPLAY_PERCENT,
            xlim=(IR_PURE_ARGON_DISPLAY_PERCENT, IR_PLOT_MAX_CONCENTRATION_PERCENT * 1.1),
            output=PROJECT_ROOT / "primary_fits" / "plots" / "plot_fit" / output_subdir / f"ArCF4_global_{line}.pdf",
            legend_kwargs={"ncol": 2},
        )
        for line in IR_LINES
    ]


def build_config(
    selection_mode: str = IR_SELECTION_MODE,
    *,
    name: str = "ArCF4_IR_primary",
    output_subdir: str = "ArCF4_IR",
) -> FitConfig:
    return FitConfig(
        name=name,
        model_name="ArCF4_infrarred",
        degrad_csv=DATA_DIR / "Primary_DegradData" / "ArCF4_IR.csv",
        datasets=build_datasets(selection_mode),
        equations=EQUATIONS,
        parameters=ir_parameters(),
        plots=build_plots(selection_mode, output_subdir=output_subdir),
        is_infrared=True,
        first_point_anchor_weight=IR_FIRST_POINT_ANCHOR_WEIGHT,
        toy_spec=ToySpec(
            n_stat=5,
            n_syst=5,
            seed=33001,
            n_jobs=-1,
            syst_sources=tuple(SystematicSource(f"line_{line}_calibration", mode="by_dataset", datasets=(line,)) for line in IR_LINES),
        ),
        table_caption=r"Parámetros del ajuste primario IR en Ar--CF$_4$.",
        table_label="tab:ArCF4_IR_primary_stat_syst",
    )


CONFIG = build_config(IR_SELECTION_MODE)


if __name__ == "__main__":
    PrimaryFitRunner(CONFIG, project_root=PROJECT_ROOT).run_all()
