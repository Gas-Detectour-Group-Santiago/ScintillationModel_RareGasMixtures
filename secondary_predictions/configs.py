from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .auxiliares.model_adapters import SecondaryModelAdapter
from .auxiliares.prediction_types import (
    BandCurveConfig,
    CombinedBandCurveConfig,
    MetadataCurveConfig,
    MetadataPlotConfig,
    MultiBandPlotConfig,
    NormalizationConfig,
    OCWBandConfig,
    OCWParameterRule,
    SecondarySelection,
)


def find_project_root(file: str | Path) -> Path:
    path = Path(file).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "data").is_dir() and (parent / "models").is_dir() and (parent / "primary_fits").is_dir():
            return parent
    return path.parents[2]


PROJECT_ROOT = find_project_root(__file__)
for folder in ("models", "primary_fits"):
    p = str(PROJECT_ROOT / folder)
    if p not in sys.path:
        sys.path.insert(0, p)

from primary_fits.ArCF4_fit import CONFIG as ARCF4_CONFIG  # noqa: E402
from primary_fits.ArCF4_IR_fit import CONFIG as ARCF4_IR_CONFIG  # noqa: E402


GARFIELD_ARCF4_POPULATIONS = PROJECT_ROOT / "data" / "Secondary_GarfieldData" / "ArCF4" / "populations" / "ArCF4_secondary.csv"
ARCF4_PAPER_DIR = PROJECT_ROOT / "data" / "Secondary_GarfieldData" / "ArCF4_paper"

# Self-contained reference folders for paper-style secondary plots.
# Each folder must contain its own populations/ArCF4_secondary.csv.
# The runner will use only that CSV for the corresponding plot/curve.
ARCF4_PAPER_REFS = {
    "gem_200mbar": ARCF4_PAPER_DIR / "gem_200mbar",
    "gem_1bar": ARCF4_PAPER_DIR / "gem_1bar",
    "gem_10bar": ARCF4_PAPER_DIR / "gem_10bar",
    "thgem_50mbar": ARCF4_PAPER_DIR / "thgem_50mbar",
    "thgem_1bar": ARCF4_PAPER_DIR / "thgem_1bar",
    "thgem_10bar": ARCF4_PAPER_DIR / "thgem_10bar",
    "electricField": ARCF4_PAPER_DIR / "electricField",
}

GARFIELD_HECF4_POPULATIONS = PROJECT_ROOT / "data" / "Secondary_GarfieldData" / "HeCF4" / "populations" / "HeCF4_secondary.csv"
X_RAY_ENERGY_ARCF4_KEV = 15.0

# Main user-facing switches for the secondary plots below.
#   "ocw_bands" -> only the hand-defined OCW band + OCW optimum line
#   "sys_stat"  -> only the propagated statistical/systematic band
#   "sum"       -> both OCW and stat.⊕syst. bands in the same figure
VISIBLE_BAND_MODE = "ocw_bands"
VISIBLE_XSCALE = "log"
VISIBLE_YSCALE = "log"
FIELD_SCAN_XSCALE = "linear"
FIELD_SCAN_YSCALE = "linear"

# Metadata/charge-balance plots.  Change these two lines if you want another
# quantity on the y-axis, e.g. PAPER_METADATA_Y = "ne", "ni", "gap_mm",
# "electric_field", "npe", ...
PAPER_METADATA_Y = "ni_minus_ne_over_ni"
PAPER_METADATA_YLABEL = r"$(n_i-n_e)/n_i$"


def _sum_components(*funcs):
    def total(params, garfield_data, concentration, pressure):
        out = None
        for func in funcs:
            value = np.asarray(func(params, garfield_data, concentration, pressure), dtype=float)
            out = value if out is None else out + value
        return out

    return total


SECONDARY_ADAPTERS = {
    "ArCF4_primary": SecondaryModelAdapter(
        fit_name="ArCF4_primary",
        garfield_csv=GARFIELD_ARCF4_POPULATIONS,
        components={
            "vis": ARCF4_CONFIG.equations["vis"],
            "uv": ARCF4_CONFIG.equations["uv"],
        },
    ),
    "ArCF4_IR_primary": SecondaryModelAdapter(
        fit_name="ArCF4_IR_primary",
        garfield_csv=GARFIELD_ARCF4_POPULATIONS,
        components={
            "696": ARCF4_IR_CONFIG.equations["696"],
            "727": ARCF4_IR_CONFIG.equations["727"],
            "750": ARCF4_IR_CONFIG.equations["750"],
            "763": ARCF4_IR_CONFIG.equations["763"],
            "772": ARCF4_IR_CONFIG.equations["772"],
            "total": _sum_components(*[ARCF4_IR_CONFIG.equations[k] for k in ("696", "727", "750", "763", "772")]),
        },
    ),
}


# Secondary Garfield normalization requested here:
#     raw_model * X_RAY_ENERGY / NORM / NPE / NE
# For the IR fit, NORM is the same Ar--CF4 primary normalization used before.
ARCF4_SECONDARY_NORM_NE = NormalizationConfig(
    mode="secondary",
    reference_fit_name="ArCF4_primary",
    output_scale=X_RAY_ENERGY_ARCF4_KEV,
    output_unit="ph/e-",
)


# Garfield selections used by the secondary plots.  The masks are still used,
# but every paper curve can now point to its own reference folder.
def _pressure_tag(pressure_bar: float) -> str:
    return str(pressure_bar).replace(".", "p")


def _paper_ref(name: str) -> Path:
    try:
        return ARCF4_PAPER_REFS[name]
    except KeyError as exc:
        raise KeyError(f"Referencia ArCF4_paper desconocida: {name!r}. Disponibles: {sorted(ARCF4_PAPER_REFS)}") from exc


def _paper_concentration_selection(
    reference_name: str,
    *,
    pressure_bar: float,
    gap_mm: float,
    gap_atol: float | None = None,
    electric_field_min: float = 0.0,
) -> SecondarySelection:
    pressure_tag = _pressure_tag(pressure_bar)
    gap_tag = str(gap_mm).replace(".", "p")
    return SecondarySelection(
        id=f"{reference_name}_gap{gap_tag}_p{pressure_tag}_concentration_scan",
        # Do not require gas_mixture == "ArCF4" here: pure CF4 rows are
        # labelled "CF4" by Analysis_secondary_garfield.py and are the real
        # 100% endpoint of the concentration scan.  Excluding them forces the
        # model to extrapolate from the last Ar--CF4 point and creates the
        # artificial collapse/spike at 100% CF4.
        gas="",
        reference_dir=_paper_ref(reference_name),
        pressure=pressure_bar,
        pressure_atol=0.026,
        gap_mm=gap_mm,
        gap_atol=(5e-4 if gap_atol is None else gap_atol),
        electric_field_min=electric_field_min,
        extra_masks={"gas_mixture": {"in": ("ArCF4", "CF4")}},
        # Keep the old paper order, but do it here in secondary_predictions:
        #   selected raw populations -> divide population columns by ne row-by-row
        #   -> model/PCHIP in concentration -> /NPE.
        # Do not touch Analysis_secondary_garfield.py for this behaviour.
        normalize_by="pre_ne",
    )




def _hecf4_concentration_selection(
    label_name: str,
    *,
    pressure_bar: float,
    gap_mm: float,
    gap_atol: float | None = None,
    electric_field_min: float = 0.0,
) -> SecondarySelection:
    pressure_tag = _pressure_tag(pressure_bar)
    gap_tag = str(gap_mm).replace(".", "p")
    return SecondarySelection(
        id=f"HeCF4_{label_name}_gap{gap_tag}_p{pressure_tag}_concentration_scan",
        # He--CF4 comparisons use the single standard HeCF4 population table,
        # not a HeCF4_paper directory.  The selection masks choose GEM/TH-GEM,
        # pressure and gap directly inside this CSV.
        gas="",
        population_csv=GARFIELD_HECF4_POPULATIONS,
        pressure=pressure_bar,
        pressure_atol=0.026,
        gap_mm=gap_mm,
        gap_atol=(5e-4 if gap_atol is None else gap_atol),
        electric_field_min=electric_field_min,
        extra_masks={"gas_mixture": {"in": ("HeCF4", "CF4")}},
        # Same choice as paper concentration scans: normalize the selected
        # Garfield populations by ne before the model interpolates them.
        normalize_by="pre_ne",
    )


def _field_scan_1pct_ne100_selection(pressure_bar: float) -> SecondarySelection:
    tag = _pressure_tag(pressure_bar)
    return SecondarySelection(
        id=f"electricField_cf4_1pct_ne100pm20_p{tag}",
        gas="ArCF4",
        reference_dir=_paper_ref("electricField"),
        pressure=pressure_bar,
        pressure_atol=0.026,
        concentration=1.0,  # Garfield CSV stores concentrations in percent.
        concentration_atol=1e-8,
        gain_min=80.0,
        gain_max=120.0,
        gain_column="ne",
        # Field scan was already behaving correctly with the usual secondary
        # normalization. Keep the ne mask and divide by ne after evaluation.
        normalize_by="ne",
    )


GEM_SELECTION = _paper_concentration_selection("gem_1bar", pressure_bar=1.0, gap_mm=0.050)
THGEM_SELECTION = _paper_concentration_selection("thgem_1bar", pressure_bar=1.0, gap_mm=0.570, gap_atol=2e-2)


# OCW/unknown-weight band for the Ar--CF4 visible emission.
# Interpretation of the old prescription:
#   P_CF3*  : envelope x0.7--x1.0, optimum x0.85
#   PAr**   : envelope x1.0--x2.0, optimum x1.5
#   PCF4*   : envelope x0.1--x1.0, optimum x0.2
#   PCF3uv  : envelope +0.0--+0.2, optimum +0.2
ARCF4_VISIBLE_OCW = OCWBandConfig(
    id="ArCF4_visible_OCW",
    label="OCW",
    use_corners=False,
    rules=(
    #    OCWParameterRule("Nnorm", low_factor=1, high_factor=1.5, optimum_factor=1.3, clip_max=1.0),
        OCWParameterRule("P_CF3_vis_dir", low_factor=0.5, high_factor=1.0, optimum_factor=0.8, clip_max=1.0),
        OCWParameterRule("P_Ar_dbleStar", low_factor=1.0, high_factor=2.0, optimum_factor=1.5, clip_max=1.0),
        OCWParameterRule("P_CF4_dir", low_factor=0.15, high_factor=1.0, optimum_factor=0.5, clip_max=1.0),
        OCWParameterRule("P_CF3_uv_dir", low_add=0.25, high_add=0.25, optimum_add=0.25, clip_max=1.0),
    ),
)

# OCW band for the UV channel.  Here the relevant ad-hoc variations are the
# CF3 UV branch and the CF4+* direct branch; this is intentionally separate
# from the VIS OCW recipe, even though both parameters live in ArCF4_primary.
ARCF4_UV_OCW = OCWBandConfig(
    id="ArCF4_UV_OCW",
    label="OCW",
    use_corners=False,
    rules=(
        OCWParameterRule("P_CF3_vis_dir", low_factor=0.5, high_factor=1.0, optimum_factor=0.8, clip_max=1.0),
        OCWParameterRule("P_Ar_dbleStar", low_factor=1.0, high_factor=2.0, optimum_factor=1.5, clip_max=1.0),
        OCWParameterRule("P_CF4_dir", low_factor=0.15, high_factor=1.0, optimum_factor=0.5, clip_max=1.0),
        OCWParameterRule("P_CF3_uv_dir", low_add=0.25, high_add=0.25, optimum_add=0.25, clip_max=1.0),
    ),
)


def _pressure_tag(pressure_bar: float) -> str:
    return str(pressure_bar).replace(".", "p")


def _visible_curve_for_paper_reference(
    reference_name: str,
    *,
    pressure_bar: float,
    x_grid: np.ndarray,
    gap_mm: float,
    gap_atol: float | None = None,
) -> BandCurveConfig:
    tag = _pressure_tag(pressure_bar)
    gap_tag = str(gap_mm).replace(".", "p")
    return BandCurveConfig(
        id=f"ArCF4_secondary_visible_{reference_name}_p{tag}_gap{gap_tag}",
        label=rf"VIS, {pressure_bar:g} bar",
        fit_name="ArCF4_primary",
        component="vis",
        pressure=pressure_bar,
        x_grid=x_grid,
        normalization=ARCF4_SECONDARY_NORM_NE,
        selection=_paper_concentration_selection(reference_name, pressure_bar=pressure_bar, gap_mm=gap_mm, gap_atol=gap_atol),
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode=VISIBLE_BAND_MODE,
        ocw_config=ARCF4_VISIBLE_OCW,
    )


def _ir_total_curve_for_paper_reference(
    reference_name: str,
    *,
    pressure_bar: float,
    x_grid: np.ndarray,
    gap_mm: float,
    gap_atol: float | None = None,
) -> BandCurveConfig:
    tag = _pressure_tag(pressure_bar)
    gap_tag = str(gap_mm).replace(".", "p")
    return BandCurveConfig(
        id=f"ArCF4_secondary_ir_total_{reference_name}_p{tag}_gap{gap_tag}",
        label=rf"IR peaks, {pressure_bar:g} bar",
        fit_name="ArCF4_IR_primary",
        component="total",
        pressure=pressure_bar,
        x_grid=x_grid,
        normalization=ARCF4_SECONDARY_NORM_NE,
        selection=_paper_concentration_selection(reference_name, pressure_bar=pressure_bar, gap_mm=gap_mm, gap_atol=gap_atol),
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode="sys_stat",
        ocw_config=None,
    )


def _vis_ir_400_800_curve_for_paper_reference(
    reference_name: str,
    *,
    pressure_bar: float,
    x_grid: np.ndarray,
    gap_mm: float,
    gap_atol: float | None = None,
) -> CombinedBandCurveConfig:
    tag = _pressure_tag(pressure_bar)
    gap_tag = str(gap_mm).replace(".", "p")
    return CombinedBandCurveConfig(
        id=f"ArCF4_secondary_400_800nm_{reference_name}_p{tag}_gap{gap_tag}",
        label=rf"{pressure_bar:g} bar",
        curves=(
            _visible_curve_for_paper_reference(reference_name, pressure_bar=pressure_bar, x_grid=x_grid, gap_mm=gap_mm, gap_atol=gap_atol),
            _ir_total_curve_for_paper_reference(reference_name, pressure_bar=pressure_bar, x_grid=x_grid, gap_mm=gap_mm, gap_atol=gap_atol),
        ),
        operation="sum",
        uncertainty_mode="quadrature",
        x_plot_factor=100.0,
        x_axis="concentration",
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode=VISIBLE_BAND_MODE,
    )


def _axis_grid_from_selection(selection: SecondarySelection, x_axis: str) -> np.ndarray:
    adapter = SECONDARY_ADAPTERS["ArCF4_primary"]
    selected = adapter.select(selection)
    if selected.empty:
        raise ValueError(f"La selección {selection.id!r} no contiene filas para construir el eje {x_axis!r}.")
    column = adapter._resolve_column(selected, x_axis, selection) or x_axis
    values = pd.to_numeric(selected[column], errors="coerce").dropna().to_numpy(dtype=float)
    values = np.unique(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"La columna {column!r} no contiene valores numéricos para {selection.id!r}.")
    return np.sort(values)


def _visible_curve_for_field_scan(pressure_bar: float) -> BandCurveConfig:
    tag = _pressure_tag(pressure_bar)
    selection = _field_scan_1pct_ne100_selection(pressure_bar)
    x_grid = _axis_grid_from_selection(selection, "electric_field")
    return BandCurveConfig(
        id=f"ArCF4_secondary_visible_vsE_cf4_1pct_p{tag}_gain100pm20_allgaps",
        label=rf"VIS, {pressure_bar:g} bar",
        fit_name="ArCF4_primary",
        component="vis",
        pressure=pressure_bar,
        x_grid=x_grid,
        normalization=ARCF4_SECONDARY_NORM_NE,
        selection=selection,
        x_plot_factor=1.0,
        x_axis="electric_field",
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode=VISIBLE_BAND_MODE,
        ocw_config=ARCF4_VISIBLE_OCW,
    )


def _ir_total_curve_for_field_scan(pressure_bar: float) -> BandCurveConfig:
    tag = _pressure_tag(pressure_bar)
    selection = _field_scan_1pct_ne100_selection(pressure_bar)
    x_grid = _axis_grid_from_selection(selection, "electric_field")
    return BandCurveConfig(
        id=f"ArCF4_secondary_ir_total_vsE_cf4_1pct_p{tag}_gain100pm20_allgaps",
        label=rf"IR peaks, {pressure_bar:g} bar",
        fit_name="ArCF4_IR_primary",
        component="total",
        pressure=pressure_bar,
        x_grid=x_grid,
        normalization=ARCF4_SECONDARY_NORM_NE,
        selection=selection,
        x_plot_factor=1.0,
        x_axis="electric_field",
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode="sys_stat",
        ocw_config=None,
    )


def _vis_ir_400_800_curve_for_field_scan(pressure_bar: float) -> CombinedBandCurveConfig:
    tag = _pressure_tag(pressure_bar)
    x_grid = _axis_grid_from_selection(_field_scan_1pct_ne100_selection(pressure_bar), "electric_field")
    return CombinedBandCurveConfig(
        id=f"ArCF4_secondary_400_800nm_vsE_cf4_1pct_p{tag}_gain100pm20_allgaps",
        label=rf"400--800 nm, {pressure_bar:g} bar",
        curves=(
            _visible_curve_for_field_scan(pressure_bar),
            _ir_total_curve_for_field_scan(pressure_bar),
        ),
        operation="sum",
        uncertainty_mode="quadrature",
        x_plot_factor=1.0,
        x_axis="electric_field",
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode=VISIBLE_BAND_MODE,
    )


def _electric_field_scan_plots(outdir: Path) -> list[MultiBandPlotConfig]:
    curves = tuple(_vis_ir_400_800_curve_for_field_scan(pressure_bar) for pressure_bar in (0.05, 1.0, 10.0))
    config = MultiBandPlotConfig(
        id="ArCF4_secondary_400_800nm_vsE_cf4_1pct_p0p05_p1p0_p10p0_gain100pm20_allgaps",
        title=r"Secondary Ar--CF$_4$ 400--800 nm vs E, 1% CF$_4$, $N_e=100\pm20$",
        curves=curves,
        xlabel=r"Electric field [kV/cm]",
        ylabel=r"Secondary 400--800 nm yield [ph/e$^-$]",
        xscale=FIELD_SCAN_XSCALE,
        yscale=FIELD_SCAN_YSCALE,
        xlim=None,
        output=outdir / "ArCF4_secondary_400_800nm_vsE_cf4_1pct_p0p05_p1p0_p10p0_gain100pm20_allgaps.pdf",
        legend_ncol=1,
        legend_loc="best",
    )
    return [config]


def _metadata_curve(reference_name: str, *, label: str, selection: SecondarySelection) -> MetadataCurveConfig:
    return MetadataCurveConfig(
        id=f"{reference_name}_{selection.id}_metadata",
        label=label,
        selection=selection,
        marker="o",
        linestyle="-",
    )


def _make_config_paper(
    *,
    y: str = PAPER_METADATA_Y,
    ylabel: str = PAPER_METADATA_YLABEL,
) -> list[MetadataPlotConfig]:
    outdir = PROJECT_ROOT / "secondary_predictions" / "plots" / "secondary_metadata"

    gem_curves = (
        _metadata_curve(
            "gem_200mbar",
            label="0.2 bar",
            selection=_paper_concentration_selection("gem_200mbar", pressure_bar=0.2, gap_mm=0.050),
        ),
        _metadata_curve(
            "gem_1bar",
            label="1 bar",
            selection=_paper_concentration_selection("gem_1bar", pressure_bar=1.0, gap_mm=0.050),
        ),
        _metadata_curve(
            "gem_10bar",
            label="10 bar",
            selection=_paper_concentration_selection("gem_10bar", pressure_bar=10.0, gap_mm=0.050),
        ),
    )

    thgem_curves = (
        _metadata_curve(
            "thgem_50mbar",
            label="0.05 bar",
            selection=_paper_concentration_selection("thgem_50mbar", pressure_bar=0.05, gap_mm=0.570, gap_atol=2e-2),
        ),
        _metadata_curve(
            "thgem_1bar",
            label="1 bar",
            selection=_paper_concentration_selection("thgem_1bar", pressure_bar=1.0, gap_mm=0.570, gap_atol=2e-2),
        ),
        _metadata_curve(
            "thgem_10bar",
            label="10 bar",
            selection=_paper_concentration_selection("thgem_10bar", pressure_bar=10.0, gap_mm=0.570, gap_atol=2e-2),
        ),
    )

    field_curves = (
        _metadata_curve(
            "electricField_0p05bar",
            label="0.05 bar",
            selection=_field_scan_1pct_ne100_selection(0.05),
        ),
        _metadata_curve(
            "electricField_1bar",
            label="1 bar",
            selection=_field_scan_1pct_ne100_selection(1.0),
        ),
        _metadata_curve(
            "electricField_10bar",
            label="10 bar",
            selection=_field_scan_1pct_ne100_selection(10.0),
        ),
    )

    return [
        MetadataPlotConfig(
            id="ArCF4_paper_metadata_GEM_concentration",
            title=r"Ar--CF$_4$ GEM charge balance",
            curves=gem_curves,
            x_axis="concentration",
            y=y,
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=ylabel,
            xscale="log",
            yscale="linear",
            xlim=(0.09, 110.0),
            output=outdir / "ArCF4_paper_metadata_GEM_concentration.pdf",
        ),
        MetadataPlotConfig(
            id="ArCF4_paper_metadata_THGEM_concentration",
            title=r"Ar--CF$_4$ TH-GEM charge balance",
            curves=thgem_curves,
            x_axis="concentration",
            y=y,
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=ylabel,
            xscale="log",
            yscale="linear",
            xlim=(0.09, 110.0),
            output=outdir / "ArCF4_paper_metadata_THGEM_concentration.pdf",
        ),
        MetadataPlotConfig(
            id="ArCF4_paper_metadata_vsE_cf4_1pct",
            title=r"Ar--CF$_4$ charge balance vs E, 1% CF$_4$, $N_e=100\pm20$",
            curves=field_curves,
            x_axis="electric_field",
            y=y,
            xlabel=r"Electric field [kV/cm]",
            ylabel=ylabel,
            xscale=FIELD_SCAN_XSCALE,
            yscale="linear",
            xlim=None,
            output=outdir / "ArCF4_paper_metadata_vsE_cf4_1pct.pdf",
        ),
    ]


# User-facing metadata plot groups.  ``config_paper`` is ready to run;
# ``config_comparation`` is intentionally empty for future experimental
# comparison overlays/configs.
config_paper: list[MetadataPlotConfig] = _make_config_paper()
config_comparation: list[MetadataPlotConfig] = []


def secondary_metadata_plots(which: str = "paper") -> list[MetadataPlotConfig]:
    key = str(which).strip().lower()
    if key in {"paper", "config_paper"}:
        return list(config_paper)
    if key in {"comparation", "comparison", "config_comparation"}:
        return list(config_comparation)
    raise ValueError("which debe ser 'paper' o 'comparation'.")


def secondary_multiband_plots() -> list[MultiBandPlotConfig]:
    x_grid = np.logspace(-3, 0, 700)  # 0.1%--100% CF4, internally as fraction.
    outdir = PROJECT_ROOT / "secondary_predictions" / "plots" / "secondary_bands"

    gem_curves = (
        _vis_ir_400_800_curve_for_paper_reference("gem_200mbar", pressure_bar=0.2, x_grid=x_grid, gap_mm=0.050),
        _vis_ir_400_800_curve_for_paper_reference("gem_1bar", pressure_bar=1.0, x_grid=x_grid, gap_mm=0.050),
        _vis_ir_400_800_curve_for_paper_reference("gem_10bar", pressure_bar=10.0, x_grid=x_grid, gap_mm=0.050),
    )
    thgem_curves = (
        _vis_ir_400_800_curve_for_paper_reference("thgem_50mbar", pressure_bar=0.05, x_grid=x_grid, gap_mm=0.570, gap_atol=2e-2),
        _vis_ir_400_800_curve_for_paper_reference("thgem_1bar", pressure_bar=1.0, x_grid=x_grid, gap_mm=0.570, gap_atol=2e-2),
        _vis_ir_400_800_curve_for_paper_reference("thgem_10bar", pressure_bar=10.0, x_grid=x_grid, gap_mm=0.570, gap_atol=2e-2),
    )

    gem_config = MultiBandPlotConfig(
        id="ArCF4_paper_400_800nm_GEM_concentration",
        title=r"Secondary Ar--CF$_4$ 400--800 nm, GEM",
        curves=gem_curves,
        xlabel=r"CF$_4$ concentration [%]",
        ylabel=r"Secondary 400--800 nm yield [ph/e$^-$]",
        xscale=VISIBLE_XSCALE,
        yscale=VISIBLE_YSCALE,
        xlim=(0.09, 110.0),
        ylim=(0.005, 2.0),
        output=outdir / "ArCF4_paper_400_800nm_GEM_concentration.pdf",
        legend_ncol=1,
        legend_loc="best",
    )
    thgem_config = MultiBandPlotConfig(
        id="ArCF4_paper_400_800nm_THGEM_concentration",
        title=r"Secondary Ar--CF$_4$ 400--800 nm, TH-GEM",
        curves=thgem_curves,
        xlabel=r"CF$_4$ concentration [%]",
        ylabel=r"Secondary 400--800 nm yield [ph/e$^-$]",
        xscale=VISIBLE_XSCALE,
        yscale=VISIBLE_YSCALE,
        xlim=(0.09, 110.0),
        ylim=(0.005, 10.0),
        output=outdir / "ArCF4_paper_400_800nm_THGEM_concentration.pdf",
        legend_ncol=1,
        legend_loc="best",
    )
    return [gem_config, thgem_config, *_electric_field_scan_plots(outdir)]


# Kept as a small template for future VIS + IR combined plots.
def arcf4_visible_plus_ir_total_curve(x_grid: np.ndarray, *, band_mode: str = "sys_stat") -> CombinedBandCurveConfig:
    vis_curve = BandCurveConfig(
        id="ArCF4_secondary_vis_gap0p050_gain100pm20",
        label="VIS",
        fit_name="ArCF4_primary",
        component="vis",
        pressure=None,
        x_grid=x_grid,
        normalization=ARCF4_SECONDARY_NORM_NE,
        selection=GEM_SELECTION,
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode=band_mode,
        ocw_config=ARCF4_VISIBLE_OCW,
    )
    ir_total_curve = BandCurveConfig(
        id="ArCF4_secondary_ir_gap0p050_gain100pm20",
        label="IR total",
        fit_name="ArCF4_IR_primary",
        component="total",
        pressure=None,
        x_grid=x_grid,
        normalization=ARCF4_SECONDARY_NORM_NE,
        selection=GEM_SELECTION,
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode="sys_stat",
    )
    return CombinedBandCurveConfig(
        id="ArCF4_secondary_vis_plus_ir_total_gap0p050_gain100pm20",
        label="VIS + IR total",
        curves=(vis_curve, ir_total_curve),
        operation="sum",
        uncertainty_mode="quadrature",
        x_plot_factor=100.0,
        x_axis="concentration",
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode=band_mode,
    )

# Backward-compatible name for older notebooks/scripts that still import
# SECONDARY_PLOTS.  Keep this lazy/empty because the ArCF4_paper reference
# CSVs may not exist yet at import time; call secondary_multiband_plots()
# after running data/Analysis_secondary_garfield.py to build the real configs.
SECONDARY_PLOTS: list[MultiBandPlotConfig] = []
