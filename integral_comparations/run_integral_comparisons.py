from __future__ import annotations

"""
Class-based integral comparisons for experimental spectra.

Default run:
    Six Ar--N2 / Ar--CF4(95/5, 1 bar) cases

    columns:     mean, C1, C2
    methods:     hardcut/trapz, gaussian_fit

The denominator is kept fixed:
    Ar--CF4 95/5, 1 bar, VIS 500-750 nm, exported raw CSV spectrum, hardcut/trapz.

Outputs:
    integral_comparations/csv/
    integral_comparations/plots/
    integral_comparations/gaussian_fit/
"""

from dataclasses import dataclass
from pathlib import Path
import sys


def _bootstrap_package_imports() -> Path:
    """Make imports work both from terminal and from notebook cells."""

    candidates: list[Path] = []

    if "__file__" in globals():
        this_file = Path(__file__).resolve()
        candidates.extend([this_file.parents[1], this_file.parent.parent])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, cwd.parent, (cwd / "..").resolve(), (cwd / "integral_comparations" / "..").resolve()])

    for candidate in candidates:
        candidate = candidate.resolve()
        package_marker = candidate / "integral_comparations" / "aux" / "integrators.py"
        if package_marker.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    return cwd


_BOOTSTRAP_ROOT = _bootstrap_package_imports()

import numpy as np
import pandas as pd

from integral_comparations.aux.integrators import IntegralConfig, integrate_spectrum
from integral_comparations.aux.paths import find_repo_root
from integral_comparations.aux.plotting import (
    GaussianFitPlotter,
    GaussianMosaicPanel,
    GaussianMosaicPlotConfig,
    GaussianPlotConfig,
    RatioGridPlotConfig,
    RatioPlotConfig,
    RatioPlotter,
)
from integral_comparations.aux.ratios import IntegralDefinition, RatioDefinition, RatioScanner, ScanConfig
from integral_comparations.aux.primary_prediction_tables import (
    AbsolutePredictionTableConfig,
    PrimaryPredictionReference,
    build_absolute_prediction_table,
    write_absolute_prediction_latex,
)
from integral_comparations.aux.n2_pure_gaussian_table import export_n2_pure_gaussian_mean_spectrum_table
from integral_comparations.aux.spectra_io import SCAN, SpectrumProvider, SpectrumSelector

# =============================================================================
# USER CONFIGURATION
# =============================================================================

ROOT_DIR = find_repo_root(__file__)
OUT_DIR = ROOT_DIR / "integral_comparations"
CSV_DIR = OUT_DIR / "csv"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"
GAUSSIAN_PLOT_DIR = OUT_DIR / "gaussian_fit"

SPECTRUM_SOURCE = "raw_csv"
SPECTRUM_UNIT = "raw"

# Common comparison.
NUMERATOR_GAS = "ArN2"
NUMERATOR_CONCENTRATION_PERCENT = SCAN
NUMERATOR_PRESSURE_BAR = SCAN
NUMERATOR_RANGE_NM = (323.0, 450.0)

DENOMINATOR_GAS = "ArCF4"
DENOMINATOR_CONCENTRATION_PERCENT = 5.0
DENOMINATOR_PRESSURE_BAR = 1.0
DENOMINATOR_SPECTRUM_COLUMN = "mean_spectrum"
DENOMINATOR_FALLBACK_COLUMNS = ("C1_spectrum", "C2_spectrum", "data(norm)")
DENOMINATOR_RANGE_NM = (500.0, 750.0)

# Scan/plot selection. None = all available numerator values.
PRESSURES_TO_PLOT = None
CONCENTRATIONS_TO_PLOT = None
CONCENTRATION_RANGE_PERCENT = None
PRESSURE_RANGE_BAR = None

# Summary point requested for the 6-row table.
# Default: pure Ar--N2 spectrum at 1 bar.
SUMMARY_PRESSURE_BAR = 1.0
SUMMARY_CONCENTRATION_PERCENT = 100.0

CSV_NAME = "ArN2_over_ArCF4_95_5_1bar_VIS_six_cases_ratio_scan.csv"
SUMMARY_CSV_NAME = "ArN2_over_ArCF4_95_5_1bar_VIS_six_cases_1bar_100pct.csv"
FIG_NAME = "ArN2_over_ArCF4_95_5_1bar_VIS_six_cases_ratio_grid.pdf"

# Optional conversion of the six Ar--N2 integral cases into absolute primary
# predictions. The default anchor is the pure N2, 1 bar, mean_spectrum hardcut
# case itself. That avoids using Ar--CF4 95/5 as a physical denominator in the
# final table: Ar--CF4 is only an internal common denominator for computing
# relative integral factors.
WRITE_ABSOLUTE_PREDICTION_TABLE = True
PRIMARY_PREDICTION_SOURCE_CSV = ROOT_DIR / "data" / "Predictions" / "primary_selected_yields_arcf4_vs_arn2_norm.csv"
PRIMARY_REFERENCE_ID = "N2_UV_N2"
PRIMARY_REFERENCE_FALLBACK_IDS: tuple[str, ...] = ()
PRIMARY_REFERENCE_TEX_LABEL = r"$Y_{\mathrm{N_2,UV}}(100\%\,\mathrm{N_2})$"

# How the primary reference is applied to the integral summary:
#   - "relative_to_anchor": absolute_i = primary_ref * ratio_i / ratio_anchor.
#   - "direct_ratio":       absolute_i = primary_ref * ratio_i.
PRIMARY_REFERENCE_SCALE_MODE = "relative_to_anchor"
PRIMARY_REFERENCE_ANCHOR_RATIO_NAME = "ArN2_mean_hardcut_over_ArCF4_95_5_VIS"
PRIMARY_REFERENCE_ANCHOR_CASE_LABEL = "mean, hardcut"

ABSOLUTE_PREDICTION_CSV_NAME = "ArN2_six_cases_scaled_to_N2_mean_hardcut_primary_norms.csv"
ABSOLUTE_PREDICTION_TEX_NAME = "ArN2_six_cases_scaled_to_N2_mean_hardcut_primary_norms.tex"
ABSOLUTE_PREDICTION_TABLE_CAPTION = (
    rf"Predicciones absolutas de Ar--N$_2$ a {SUMMARY_PRESSURE_BAR:g} bar y "
    rf"{SUMMARY_CONCENTRATION_PERCENT:g}\% N$_2$. El caso mean+hardcut se fija a la "
    r"predicción primaria de N$_2$ puro; el resto de casos se obtiene mediante "
    r"factores integrales relativos al mismo espectro."
)
ABSOLUTE_PREDICTION_TABLE_LABEL = "tab:arn2_six_cases_scaled_to_n2_mean_hardcut_primary_norms"
ABSOLUTE_PREDICTION_INCLUDE_RATIO_COLUMN = False

WRITE_N2_PURE_GAUSSIAN_MEAN_TABLE = True
N2_PURE_PRIMARY_PREDICTION_CSV = ROOT_DIR / "data" / "Predictions" / "N2_pure_predictions_by_incident_type.csv"
N2_PURE_GAUSSIAN_CSV_NAME = "N2_pure_predictions_gaussian_mean_spectrum.csv"
N2_PURE_GAUSSIAN_TEX_NAME = "N2_pure_predictions_gaussian_mean_spectrum.tex"
N2_PURE_GAUSSIAN_TABLE_CAPTION = (
    r"Predicciones de N$_2$ puro para distintas entradas de Degrad tras aplicar "
    r"la corrección integral gaussiana del espectro medio."
)
N2_PURE_GAUSSIAN_TABLE_LABEL = "tab:n2_pure_predictions_gaussian_mean_spectrum"

WRITE_GAUSSIAN_DIAGNOSTICS = True

# Gaussian diagnostic output.
#   - "mosaic": one PDF mosaic per gaussian case and selected pressure.
#   - "individual": one PDF per fitted spectrum.
#   - "both": write both outputs.
GAUSSIAN_DIAGNOSTIC_MODE = "mosaic"

# None = all available values. By default only the 1 bar spectra are drawn.
GAUSSIAN_DIAGNOSTIC_PRESSURES = (1.0,)

# Select the concentrations to place in the mosaic, in this order.
# None = all concentrations available at the selected pressure(s).
GAUSSIAN_DIAGNOSTIC_CONCENTRATIONS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)

# Select which gaussian ratio panels to diagnose. None = all gaussian-fit cases.
GAUSSIAN_DIAGNOSTIC_RATIO_NAMES = None

# Gaussian peak profile used by the gaussian integral.
# Available profiles are defined in PEAK_LIBRARY below.
# The default five-peak Ar--N2 UV profile follows the approximate centres seen
# in the extracted spectra: 336, 356, 380, 410 and 430 nm.
# Keeping GAUSSIAN_CENTER_MODE = "fixed" prevents the last two components from
# drifting into the main 336--380 nm structure.
GAUSSIAN_PEAK_PROFILE = "ArN2_UV_5"

# Because the nominal band-head wavelengths are known, fixed centres are the
# default. Change to "bounded" or "shared_shift" if you want to absorb small
# calibration shifts in the experimental wavelength axis.
GAUSSIAN_CENTER_MODE = "fixed"
GAUSSIAN_CENTER_TOLERANCE_NM = 1.5
GAUSSIAN_WIDTH_MODE = "shared"
GAUSSIAN_SIGMA_NM = 3.0
GAUSSIAN_SIGMA_BOUNDS_NM = (0.5, 12.0)
GAUSSIAN_BASELINE_MODE = "constant"
GAUSSIAN_INTEGRATE_BASELINE = False

# Number of panels inside each mosaic. A 3x3 layout gives nine concentrations.
# If more concentrations are selected, additional mosaic pages are created.
GAUSSIAN_MOSAIC_NROWS = 3
GAUSSIAN_MOSAIC_NCOLS = 3
GAUSSIAN_MOSAIC_PANEL_SIZE = (4.0, 2.8)
GAUSSIAN_MOSAIC_SPLIT_OVERFLOW = True

# =============================================================================
# CASE CONFIGURATION
# =============================================================================

PEAK_LIBRARY: dict[str, tuple[tuple[str, float], ...]] = {
    # Approximate Ar--N2 UV band centres used for the Gaussian decomposition.
    # These are deliberately fixed by default. The fifth component is placed
    # after the 410 nm structure, around 430 nm, so it cannot collapse into the
    # first three strong bands.
    "ArN2_UV_4": (
        ("N2_336", 336.0),
        ("N2_356", 356.0),
        ("N2_380", 380.0),
        ("N2_410", 410.0),
    ),
    "ArN2_UV_5": (
    ("N2_336", 336.0),
    ("N2_356", 356.0),
    ("N2_380", 380.0),
    ("N2_410", 403.0),
    ("N2_430", 431.0),
    ),
    # Extra diagnostic profiles kept only for manual checks.
    "ArN2_UV_6": (
        ("N2_336", 336.0),
        ("N2_356", 356.0),
        ("N2_380", 380.0),
        ("N2plus_391", 391.4),
        ("N2_410", 410.0),
        ("N2_430", 430.0),
    ),
    "ArN2_UV_7": (
        ("N2_336", 336.0),
        ("N2_356", 356.0),
        ("N2_380", 380.0),
        ("N2plus_391", 391.4),
        ("N2_410", 410.0),
        ("N2_424", 424.2),
        ("N2_430", 430.0),
    ),
    # Backward-compatible alias of the default five-peak profile.
    "ArN2_UV": (
        ("N2_336", 336.0),
        ("N2_356", 356.0),
        ("N2_380", 380.0),
        ("N2_410", 410.0),
        ("N2_430", 430.0),
    ),
}



@dataclass(frozen=True)
class SpectrumColumnCase:
    key: str
    label: str
    spectrum_column: str
    fallbacks: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntegrationCase:
    key: str
    label: str
    method: str
    gaussian_profile_name: str | None = None
    gaussian_center_mode: str = "shared_shift"
    gaussian_center_tolerance_nm: float = 1.0
    gaussian_width_mode: str = "shared"
    gaussian_sigma_nm: float | tuple[float, ...] = 3.0
    gaussian_sigma_bounds_nm: tuple[float, float] = (0.5, 12.0)
    gaussian_baseline_mode: str = "constant"
    gaussian_integrate_baseline: bool = False


SPECTRUM_CASES = (
    SpectrumColumnCase("mean", "mean", "mean_spectrum", ("spectrum_new_cal", "spectrum_old_cal")),
    SpectrumColumnCase("C1", "C1", "C1_spectrum", ("C1", "mean_spectrum", "spectrum_new_cal")),
    SpectrumColumnCase("C2", "C2", "C2_spectrum", ("C2", "mean_spectrum", "spectrum_old_cal")),
)

INTEGRATION_CASES = (
    IntegrationCase("hardcut", "hardcut", "trapz"),
    IntegrationCase(
        "gaussian",
        "gaussian",
        "gaussian_fit",
        gaussian_profile_name=GAUSSIAN_PEAK_PROFILE,
        gaussian_center_mode=GAUSSIAN_CENTER_MODE,
        gaussian_center_tolerance_nm=GAUSSIAN_CENTER_TOLERANCE_NM,
        gaussian_width_mode=GAUSSIAN_WIDTH_MODE,
        gaussian_sigma_nm=GAUSSIAN_SIGMA_NM,
        gaussian_sigma_bounds_nm=GAUSSIAN_SIGMA_BOUNDS_NM,
        gaussian_baseline_mode=GAUSSIAN_BASELINE_MODE,
        gaussian_integrate_baseline=GAUSSIAN_INTEGRATE_BASELINE,
    ),
)


@dataclass(frozen=True)
class GaussianProfile:
    names: tuple[str, ...]
    centers_nm: tuple[float, ...]


def get_gaussian_profile(name: str | None) -> GaussianProfile:
    if not name:
        return GaussianProfile(names=(), centers_nm=())
    if name not in PEAK_LIBRARY:
        raise KeyError(f"Unknown Gaussian peak profile {name!r}. Available: {sorted(PEAK_LIBRARY)}")
    entries = PEAK_LIBRARY[name]
    return GaussianProfile(
        names=tuple(entry[0] for entry in entries),
        centers_nm=tuple(float(entry[1]) for entry in entries),
    )


def build_integral_config(case: IntegrationCase) -> IntegralConfig:
    profile = get_gaussian_profile(case.gaussian_profile_name)
    return IntegralConfig(
        wavelength_range_nm=NUMERATOR_RANGE_NM,
        method=case.method,  # type: ignore[arg-type]
        gaussian_centers_nm=profile.centers_nm,
        gaussian_peak_names=profile.names,
        gaussian_center_mode=case.gaussian_center_mode,  # type: ignore[arg-type]
        gaussian_center_tolerance_nm=case.gaussian_center_tolerance_nm,
        gaussian_width_mode=case.gaussian_width_mode,  # type: ignore[arg-type]
        gaussian_sigma_nm=case.gaussian_sigma_nm,
        gaussian_sigma_bounds_nm=case.gaussian_sigma_bounds_nm,
        gaussian_baseline_mode=case.gaussian_baseline_mode,  # type: ignore[arg-type]
        gaussian_integrate_baseline=case.gaussian_integrate_baseline,
    )


def build_denominator_integral_config() -> IntegralConfig:
    return IntegralConfig(
        wavelength_range_nm=DENOMINATOR_RANGE_NM,
        method="trapz",
    )


def make_ratio_definitions() -> tuple[list[RatioDefinition], dict[str, str]]:
    denominator = IntegralDefinition(
        name=f"{DENOMINATOR_GAS}_{DENOMINATOR_RANGE_NM[0]:g}_{DENOMINATOR_RANGE_NM[1]:g}nm_hardcut",
        selector=SpectrumSelector(
            gas=DENOMINATOR_GAS,
            concentration_percent=DENOMINATOR_CONCENTRATION_PERCENT,
            pressure_bar=DENOMINATOR_PRESSURE_BAR,
            spectrum_column=DENOMINATOR_SPECTRUM_COLUMN,
            column_fallbacks=DENOMINATOR_FALLBACK_COLUMNS,
            source=SPECTRUM_SOURCE,
            unit=SPECTRUM_UNIT,
            include_ir_yield=True,
        ),
        integral=build_denominator_integral_config(),
    )

    ratios: list[RatioDefinition] = []
    titles: dict[str, str] = {}

    for spectrum_case in SPECTRUM_CASES:
        for integration_case in INTEGRATION_CASES:
            ratio_name = f"ArN2_{spectrum_case.key}_{integration_case.key}_over_ArCF4_95_5_VIS"
            titles[ratio_name] = f"{spectrum_case.label}, {integration_case.label}"

            numerator = IntegralDefinition(
                name=f"ArN2_{spectrum_case.key}_{integration_case.key}_{NUMERATOR_RANGE_NM[0]:g}_{NUMERATOR_RANGE_NM[1]:g}nm",
                selector=SpectrumSelector(
                    gas=NUMERATOR_GAS,
                    concentration_percent=NUMERATOR_CONCENTRATION_PERCENT,
                    pressure_bar=NUMERATOR_PRESSURE_BAR,
                    spectrum_column=spectrum_case.spectrum_column,
                    column_fallbacks=spectrum_case.fallbacks,
                    source=SPECTRUM_SOURCE,
                    unit=SPECTRUM_UNIT,
                    include_ir_yield=True,
                ),
                integral=build_integral_config(integration_case),
            )

            ratios.append(
                RatioDefinition(
                    name=ratio_name,
                    numerator=numerator,
                    denominator=denominator,
                    description=(
                        f"{spectrum_case.label} spectrum, {integration_case.label} integral: "
                        f"ArN2 {NUMERATOR_RANGE_NM[0]:g}-{NUMERATOR_RANGE_NM[1]:g} nm / "
                        f"ArCF4 95/5, 1 bar, {DENOMINATOR_RANGE_NM[0]:g}-{DENOMINATOR_RANGE_NM[1]:g} nm"
                    ),
                )
            )

    return ratios, titles


def build_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"].eq("ok")].copy() if "status" in results.columns else results.copy()
    if ok.empty:
        return ok

    p = pd.to_numeric(ok["pressure_bar"], errors="coerce")
    c = pd.to_numeric(ok["concentration_percent"], errors="coerce")
    summary = ok[
        np.isclose(p, SUMMARY_PRESSURE_BAR)
        & np.isclose(c, SUMMARY_CONCENTRATION_PERCENT)
    ].copy()

    if summary.empty:
        return summary

    summary["case"] = summary["ratio_name"].astype(str)
    summary["method"] = summary["numerator_method"].astype(str)
    summary["spectrum_column"] = summary["numerator_spectrum_column"].astype(str)

    columns = [
        "case",
        "spectrum_column",
        "method",
        "pressure_bar",
        "concentration_percent",
        "ratio",
        "numerator_integral",
        "denominator_integral",
        "numerator_range_nm",
        "denominator_range_nm",
        "numerator_n_points",
        "denominator_n_points",
    ]
    return summary[columns].sort_values(["spectrum_column", "method"]).reset_index(drop=True)


def gaussian_integral_is_used(ratios: list[RatioDefinition]) -> bool:
    return any(
        ratio.numerator.integral.method == "gaussian_fit"
        or ratio.denominator.integral.method == "gaussian_fit"
        for ratio in ratios
    )


def _selected_for_gaussian_diagnostic(pressure: float, concentration: float) -> bool:
    if GAUSSIAN_DIAGNOSTIC_PRESSURES is not None:
        if not any(np.isclose(pressure, p) for p in GAUSSIAN_DIAGNOSTIC_PRESSURES):
            return False
    if GAUSSIAN_DIAGNOSTIC_CONCENTRATIONS is not None:
        if not any(np.isclose(concentration, c) for c in GAUSSIAN_DIAGNOSTIC_CONCENTRATIONS):
            return False
    return True


def _gaussian_diagnostic_mode_includes(kind: str) -> bool:
    mode = str(GAUSSIAN_DIAGNOSTIC_MODE).strip().lower()
    return mode == kind or mode == "both"


def _selected_ratio_for_gaussian_diagnostic(ratio_name: str) -> bool:
    if GAUSSIAN_DIAGNOSTIC_RATIO_NAMES is None:
        return True
    return ratio_name in set(str(name) for name in GAUSSIAN_DIAGNOSTIC_RATIO_NAMES)


def _ordered_gaussian_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    rows = rows.copy()
    rows["_concentration_sort"] = pd.to_numeric(rows["concentration_percent"], errors="coerce")

    if GAUSSIAN_DIAGNOSTIC_CONCENTRATIONS is None:
        return rows.sort_values("_concentration_sort").drop(columns="_concentration_sort")

    ordered_parts: list[pd.DataFrame] = []
    for concentration in GAUSSIAN_DIAGNOSTIC_CONCENTRATIONS:
        c = rows["_concentration_sort"].to_numpy(dtype=float)
        match = rows[np.isclose(c, float(concentration))].copy()
        if not match.empty:
            ordered_parts.append(match.iloc[[0]])

    if not ordered_parts:
        return rows.iloc[0:0].drop(columns="_concentration_sort")

    return pd.concat(ordered_parts, ignore_index=True).drop(columns="_concentration_sort")


def _chunk_rows(rows: pd.DataFrame, capacity: int) -> list[pd.DataFrame]:
    capacity = max(1, int(capacity))
    if not GAUSSIAN_MOSAIC_SPLIT_OVERFLOW:
        return [rows.iloc[:capacity].copy()]
    return [rows.iloc[i : i + capacity].copy() for i in range(0, len(rows), capacity)]


def write_gaussian_diagnostics(
    provider: SpectrumProvider,
    ratios: list[RatioDefinition],
    results: pd.DataFrame,
    ratio_titles: dict[str, str],
) -> None:
    if results.empty:
        return

    if _gaussian_diagnostic_mode_includes("individual"):
        write_individual_gaussian_diagnostics(provider, ratios, results, ratio_titles)

    if _gaussian_diagnostic_mode_includes("mosaic"):
        write_gaussian_diagnostic_mosaics(provider, ratios, results, ratio_titles)


def write_gaussian_diagnostic_mosaics(
    provider: SpectrumProvider,
    ratios: list[RatioDefinition],
    results: pd.DataFrame,
    ratio_titles: dict[str, str],
) -> None:
    plotter = GaussianFitPlotter(use_science_style=True, use_grid=False)
    ratio_by_name = {ratio.name: ratio for ratio in ratios}
    ok = results[results["status"].eq("ok")].copy() if "status" in results.columns else results.copy()
    if ok.empty:
        return

    capacity = max(1, int(GAUSSIAN_MOSAIC_NROWS) * int(GAUSSIAN_MOSAIC_NCOLS))

    for ratio_name in ok["ratio_name"].dropna().astype(str).unique():
        if not _selected_ratio_for_gaussian_diagnostic(ratio_name):
            continue
        ratio = ratio_by_name.get(ratio_name)
        if ratio is None:
            continue

        for role, definition in (("numerator", ratio.numerator), ("denominator", ratio.denominator)):
            if definition.integral.method != "gaussian_fit":
                continue

            subset = ok[ok["ratio_name"].astype(str).eq(ratio_name)].copy()
            if role == "numerator":
                subset["pressure_bar"] = pd.to_numeric(subset["numerator_pressure_bar"], errors="coerce")
                subset["concentration_percent"] = pd.to_numeric(subset["numerator_concentration_percent"], errors="coerce")
                subset["spectrum_column_used"] = subset["numerator_spectrum_column"].astype(str)
            else:
                subset["pressure_bar"] = pd.to_numeric(subset["denominator_pressure_bar"], errors="coerce")
                subset["concentration_percent"] = pd.to_numeric(subset["denominator_concentration_percent"], errors="coerce")
                subset["spectrum_column_used"] = subset["denominator_spectrum_column"].astype(str)

            subset = subset[np.isfinite(subset["pressure_bar"]) & np.isfinite(subset["concentration_percent"])]
            if subset.empty:
                continue

            for pressure in np.sort(subset["pressure_bar"].astype(float).unique()):
                if GAUSSIAN_DIAGNOSTIC_PRESSURES is not None:
                    if not any(np.isclose(pressure, float(p)) for p in GAUSSIAN_DIAGNOSTIC_PRESSURES):
                        continue

                pressure_rows = subset[np.isclose(subset["pressure_bar"].astype(float), float(pressure))].copy()
                pressure_rows = _ordered_gaussian_rows(pressure_rows)
                if pressure_rows.empty:
                    continue

                for page_index, page_rows in enumerate(_chunk_rows(pressure_rows, capacity), start=1):
                    panels: list[GaussianMosaicPanel] = []
                    for _, row in page_rows.iterrows():
                        concentration = float(row["concentration_percent"])
                        spectrum_column = str(row["spectrum_column_used"])

                        selector = definition.selector.resolved(concentration_percent=concentration, pressure_bar=float(pressure))
                        selector = SpectrumSelector(
                            gas=selector.gas,
                            concentration_percent=selector.concentration_percent,
                            pressure_bar=selector.pressure_bar,
                            spectrum_column=spectrum_column,
                            source=selector.source,
                            unit=selector.unit,
                            include_ir_yield=selector.include_ir_yield,
                            clip_negative=selector.clip_negative,
                            column_fallbacks=selector.column_fallbacks,
                        )

                        spectrum = provider.load(selector)
                        fit_result = integrate_spectrum(
                            spectrum.wavelength_nm,
                            spectrum.intensity,
                            definition.integral,
                        )
                        if fit_result.plot_payload is None:
                            continue

                        integral_value = fit_result.plot_payload.get("integral_value", fit_result.value)
                        panels.append(
                            GaussianMosaicPanel(
                                result=fit_result,
                                title=f"c={concentration:g}% | I={float(integral_value):.2e}",
                            )
                        )

                    if not panels:
                        continue

                    page_suffix = f"_page{page_index:02d}" if len(pressure_rows) > capacity else ""
                    filename = (
                        f"{_safe_filename(ratio_name)}_{_safe_filename(role)}_"
                        f"P{float(pressure):g}bar_mosaic{page_suffix}.pdf"
                    )
                    out_path = GAUSSIAN_PLOT_DIR / filename
                    plotter.plot_mosaic(
                        panels,
                        out_path,
                        GaussianMosaicPlotConfig(
                            title=(
                                f"{ratio_titles.get(ratio_name, ratio_name)} | {role}, "
                                f"P={float(pressure):g} bar"
                            ),
                            ylabel=SPECTRUM_UNIT,
                            nrows=int(GAUSSIAN_MOSAIC_NROWS),
                            ncols=int(GAUSSIAN_MOSAIC_NCOLS),
                            panel_size=tuple(float(x) for x in GAUSSIAN_MOSAIC_PANEL_SIZE),
                            sharey=False,
                        ),
                    )


def write_individual_gaussian_diagnostics(
    provider: SpectrumProvider,
    ratios: list[RatioDefinition],
    results: pd.DataFrame,
    ratio_titles: dict[str, str],
) -> None:
    plotter = GaussianFitPlotter(use_science_style=True, use_grid=False)
    ratio_by_name = {ratio.name: ratio for ratio in ratios}
    done: set[tuple[object, ...]] = set()

    ok = results[results["status"].eq("ok")].copy() if "status" in results.columns else results.copy()
    for _, row in ok.iterrows():
        ratio_name = str(row["ratio_name"])
        if not _selected_ratio_for_gaussian_diagnostic(ratio_name):
            continue
        ratio = ratio_by_name.get(ratio_name)
        if ratio is None:
            continue

        for role, definition in (("numerator", ratio.numerator), ("denominator", ratio.denominator)):
            if definition.integral.method != "gaussian_fit":
                continue

            if role == "numerator":
                concentration = float(row["numerator_concentration_percent"])
                pressure = float(row["numerator_pressure_bar"])
                spectrum_column = str(row["numerator_spectrum_column"])
            else:
                concentration = float(row["denominator_concentration_percent"])
                pressure = float(row["denominator_pressure_bar"])
                spectrum_column = str(row["denominator_spectrum_column"])

            if not _selected_for_gaussian_diagnostic(pressure, concentration):
                continue

            selector = definition.selector.resolved(concentration_percent=concentration, pressure_bar=pressure)
            selector = SpectrumSelector(
                gas=selector.gas,
                concentration_percent=selector.concentration_percent,
                pressure_bar=selector.pressure_bar,
                spectrum_column=spectrum_column,
                source=selector.source,
                unit=selector.unit,
                include_ir_yield=selector.include_ir_yield,
                clip_negative=selector.clip_negative,
                column_fallbacks=selector.column_fallbacks,
            )

            key = (
                ratio_name,
                role,
                selector.gas,
                selector.concentration_percent,
                selector.pressure_bar,
                selector.spectrum_column,
                definition.integral.method,
                definition.integral.wavelength_range_nm,
            )
            if key in done:
                continue
            done.add(key)

            spectrum = provider.load(selector)
            fit_result = integrate_spectrum(
                spectrum.wavelength_nm,
                spectrum.intensity,
                definition.integral,
            )
            if fit_result.plot_payload is None:
                continue

            filename = (
                f"{_safe_filename(role)}_{selector.gas}_P{selector.pressure_bar:g}bar_"
                f"c{selector.concentration_percent:g}pct_{_safe_filename(selector.spectrum_column)}_"
                f"{definition.integral.wavelength_range_nm[0]:g}_{definition.integral.wavelength_range_nm[1]:g}nm.pdf"
            )
            out_path = GAUSSIAN_PLOT_DIR / "individual" / _safe_filename(ratio_name) / filename
            plotter.plot(
                fit_result,
                out_path,
                GaussianPlotConfig(
                    title=(
                        f"{ratio_titles.get(ratio_name, ratio_name)} | {role}: {selector.gas}, "
                        f"P={selector.pressure_bar:g} bar, c={selector.concentration_percent:g}%, "
                        f"{selector.spectrum_column}"
                    ),
                    ylabel=SPECTRUM_UNIT,
                ),
            )

def _safe_filename(name: str) -> str:
    keep = []
    for char in str(name):
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "file"


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GAUSSIAN_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    provider = SpectrumProvider(ROOT_DIR)
    scanner = RatioScanner(provider)

    scan = ScanConfig(
        pressures_bar=None if PRESSURES_TO_PLOT is None else tuple(float(p) for p in PRESSURES_TO_PLOT),
        concentrations_percent=None if CONCENTRATIONS_TO_PLOT is None else tuple(float(c) for c in CONCENTRATIONS_TO_PLOT),
        concentration_range_percent=CONCENTRATION_RANGE_PERCENT,
        pressure_range_bar=PRESSURE_RANGE_BAR,
    )

    ratios, ratio_titles = make_ratio_definitions()
    results = scanner.compute(ratios, scan)
    if results.empty:
        raise RuntimeError("No integral ratios were computed. Check spectrum columns, pressure and concentration filters.")

    csv_path = CSV_DIR / CSV_NAME
    results.to_csv(csv_path, index=False)

    ok = results[results["status"].eq("ok")] if "status" in results.columns else results
    invalid = results[results["status"].ne("ok")] if "status" in results.columns else results.iloc[0:0]

    print("\nIntegral comparison results:")
    if ok.empty:
        print("No valid finite ratios were obtained. Check invalid rows.")
    else:
        print(ok.to_string(index=False))

    if not invalid.empty:
        invalid_csv_path = CSV_DIR / CSV_NAME.replace(".csv", "_invalid_rows.csv")
        invalid.to_csv(invalid_csv_path, index=False)
        print("\nRows with invalid ratios/errors:")
        print(invalid.to_string(index=False))
        print(f"Saved invalid rows: {invalid_csv_path.relative_to(ROOT_DIR)}")

    summary = build_summary_table(results)
    summary_csv_path = CSV_DIR / SUMMARY_CSV_NAME
    summary.to_csv(summary_csv_path, index=False)

    print(f"\nSummary at {SUMMARY_PRESSURE_BAR:g} bar and {SUMMARY_CONCENTRATION_PERCENT:g}% N2:")
    if summary.empty:
        print("No valid rows found for the requested summary point.")
    else:
        print(summary.to_string(index=False))

    absolute_csv_path: Path | None = None
    absolute_tex_path: Path | None = None
    if WRITE_ABSOLUTE_PREDICTION_TABLE and not summary.empty:
        absolute_table, absolute_metadata = build_absolute_prediction_table(
            summary,
            PrimaryPredictionReference(
                csv_path=PRIMARY_PREDICTION_SOURCE_CSV,
                reference_id=PRIMARY_REFERENCE_ID,
                fallback_ids=PRIMARY_REFERENCE_FALLBACK_IDS,
                forced_tex_label=PRIMARY_REFERENCE_TEX_LABEL,
                scale_mode=PRIMARY_REFERENCE_SCALE_MODE,  # type: ignore[arg-type]
                anchor_ratio_name=PRIMARY_REFERENCE_ANCHOR_RATIO_NAME,
                anchor_case_label=PRIMARY_REFERENCE_ANCHOR_CASE_LABEL,
            ),
            ratio_titles=ratio_titles,
        )
        if not absolute_table.empty:
            absolute_csv_path = CSV_DIR / ABSOLUTE_PREDICTION_CSV_NAME
            absolute_tex_path = TABLE_DIR / ABSOLUTE_PREDICTION_TEX_NAME
            absolute_table.to_csv(absolute_csv_path, index=False)
            write_absolute_prediction_latex(
                absolute_table,
                absolute_tex_path,
                AbsolutePredictionTableConfig(
                    caption=ABSOLUTE_PREDICTION_TABLE_CAPTION,
                    label=ABSOLUTE_PREDICTION_TABLE_LABEL,
                    unit=str(absolute_metadata.get("reference_unit", "ph/MeV")),
                    include_ratio_column=ABSOLUTE_PREDICTION_INCLUDE_RATIO_COLUMN,
                ),
            )
            print("\nAbsolute ArN2 predictions from N2 primary anchor:")
            print(
                absolute_table[
                    ["case", "scale_factor", "value_arcf4_norm", "value_arn2_norm"]
                ].to_string(index=False)
            )
        else:
            print("\nNo absolute prediction table was written: empty table after filtering.")

    n2_pure_gaussian_csv_path: Path | None = None
    n2_pure_gaussian_tex_path: Path | None = None
    if WRITE_N2_PURE_GAUSSIAN_MEAN_TABLE and not summary.empty:
        try:
            gaussian_n2_pure = export_n2_pure_gaussian_mean_spectrum_table(
                primary_prediction_csv=N2_PURE_PRIMARY_PREDICTION_CSV,
                summary_ratio_csv=summary_csv_path,
                output_csv=CSV_DIR / N2_PURE_GAUSSIAN_CSV_NAME,
                output_tex=TABLE_DIR / N2_PURE_GAUSSIAN_TEX_NAME,
                caption=N2_PURE_GAUSSIAN_TABLE_CAPTION,
                label=N2_PURE_GAUSSIAN_TABLE_LABEL,
            )
            n2_pure_gaussian_csv_path = CSV_DIR / N2_PURE_GAUSSIAN_CSV_NAME
            n2_pure_gaussian_tex_path = TABLE_DIR / N2_PURE_GAUSSIAN_TEX_NAME
            print("\nN2 pure predictions with mean-spectrum Gaussian correction:")
            print(
                gaussian_n2_pure[
                    ["id", "gaussian_over_hardcut_scale", "value_arcf4_norm", "value_arn2_norm"]
                ].to_string(index=False)
            )
        except Exception as exc:
            print(f"\nNo N2 pure gaussian table was written: {exc}")

    if ok.empty:
        print(f"\nSaved CSV: {csv_path.relative_to(ROOT_DIR)}")
        print(f"Saved summary CSV: {summary_csv_path.relative_to(ROOT_DIR)}")
        if absolute_csv_path is not None:
            print(f"Saved absolute prediction CSV: {absolute_csv_path.relative_to(ROOT_DIR)}")
        if absolute_tex_path is not None:
            print(f"Saved absolute prediction table: {absolute_tex_path.relative_to(ROOT_DIR)}")
        if n2_pure_gaussian_csv_path is not None:
            print(f"Saved N2 pure gaussian CSV: {n2_pure_gaussian_csv_path.relative_to(ROOT_DIR)}")
        if n2_pure_gaussian_tex_path is not None:
            print(f"Saved N2 pure gaussian table: {n2_pure_gaussian_tex_path.relative_to(ROOT_DIR)}")
        return

    plotter = RatioPlotter(use_science_style=True, use_grid=False)
    fig_path = PLOT_DIR / FIG_NAME
    plotter.plot_grid_by_pressure(
        results,
        fig_path,
        RatioGridPlotConfig(
            title=(
                r"Ar--N$_2$ UV integral over fixed Ar--CF$_4$ 95/5 VIS reference"
            ),
            xlabel=r"N$_2$ concentration [\%]",
            ylabel=(
                rf"$\int_{{{NUMERATOR_RANGE_NM[0]:g}}}^{{{NUMERATOR_RANGE_NM[1]:g}}} I_{{\mathrm{{Ar-N_2}}}}\,d\lambda$ / "
                rf"$\int_{{{DENOMINATOR_RANGE_NM[0]:g}}}^{{{DENOMINATOR_RANGE_NM[1]:g}}} I_{{\mathrm{{Ar-CF_4}}}}\,d\lambda$"
            ),
            pressures_bar=scan.pressures_bar,
            concentration_range_percent=scan.concentration_range_percent,
            ratio_names=tuple(ratio.name for ratio in ratios),
            ratio_titles=ratio_titles,
            xscale="auto",
            yscale="auto",
            legend_title=rf"$P_{{\mathrm{{Ar-N_2}}}}$",
            ncols=3,
            sharey=True,
        ),
    )

    if WRITE_GAUSSIAN_DIAGNOSTICS and gaussian_integral_is_used(ratios):
        write_gaussian_diagnostics(provider, ratios, results, ratio_titles)

    print(f"\nSaved CSV: {csv_path.relative_to(ROOT_DIR)}")
    print(f"Saved summary CSV: {summary_csv_path.relative_to(ROOT_DIR)}")
    if absolute_csv_path is not None:
        print(f"Saved absolute prediction CSV: {absolute_csv_path.relative_to(ROOT_DIR)}")
    if absolute_tex_path is not None:
        print(f"Saved absolute prediction table: {absolute_tex_path.relative_to(ROOT_DIR)}")
    if n2_pure_gaussian_csv_path is not None:
        print(f"Saved N2 pure gaussian CSV: {n2_pure_gaussian_csv_path.relative_to(ROOT_DIR)}")
    if n2_pure_gaussian_tex_path is not None:
        print(f"Saved N2 pure gaussian table: {n2_pure_gaussian_tex_path.relative_to(ROOT_DIR)}")
    print(f"Saved figure: {fig_path.relative_to(ROOT_DIR)}")
    if WRITE_GAUSSIAN_DIAGNOSTICS and gaussian_integral_is_used(ratios):
        print(f"Saved gaussian diagnostics under: {GAUSSIAN_PLOT_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
