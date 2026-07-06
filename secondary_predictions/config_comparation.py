from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .auxiliares.prediction_types import (
    BandCurveConfig,
    ExperimentalSeries,
    MultiBandPlotConfig,
    OCWBandConfig,
    SecondarySelection,
)
from .configs import (
    ARCF4_IR_CONFIG,
    ARCF4_IR_OCW,
    ARCF4_SECONDARY_NORM_NE,
    ARCF4_UV_OCW,
    ARCF4_VISIBLE_OCW,
    PROJECT_ROOT,
    VISIBLE_BAND_MODE,
    VISIBLE_XSCALE,
    VISIBLE_YSCALE,
    _hecf4_concentration_selection,
    _paper_concentration_selection,
)


CONFIG_NAME = "config_comparation"

# Keep the same band philosophy as the paper plots, but use the requested
# OCW prescription for the UV comparison too:
#   - visible: VIS OCW envelope + OCW optimum line;
#   - UV: UV OCW envelope (P_CF3_uv_dir +0.25 and P_CF4_dir variation);
#   - IR: propagated stat.⊕syst. envelope around the secondary optimum,
#     where all Ar--CF4 IR optical-conversion weights are doubled.
COMPARATION_VISIBLE_BAND_MODE = VISIBLE_BAND_MODE
COMPARATION_UV_BAND_MODE = "ocw_bands"
COMPARATION_IR_BAND_MODE = "sum"

X_GRID = np.logspace(-3, 0, 700)  # 0.1%--100% CF4, internally as fraction.
OUTDIR = PROJECT_ROOT / "secondary_predictions" / "plots" / "secondary_comparation"
TABLES_DIR = PROJECT_ROOT / "data" / "Tables"


def _sum_line_dicts(values: dict[int, float], errors: dict[int, float]) -> tuple[float, float]:
    y = float(sum(values.values()))
    yerr = float(np.sqrt(sum(float(e) ** 2 for e in errors.values())))
    return y, yerr


# -----------------------------------------------------------------------------
# Experimental points taken from the latest LIP comparison scripts.
# Coimbra points are drawn as hollow markers; Florian points as filled markers.
# -----------------------------------------------------------------------------
FLORIAN_INDIRECT_SCALE_GROUP = "florian_indirect"

# Ar--CF4, LIP (direct), GEM 0.050 mm, 1 bar.
_ARCF4_VIS_DIRECT_X = np.array([5.0, 10.0, 67.0, 100.0])
_ARCF4_VIS_DIRECT_Y = np.array([0.38287151, 0.38966203, 0.2802068, 0.09335376])
_ARCF4_VIS_DIRECT_ERR = np.array([
    0.11515380887765425,
    0.1103105398012753,
    0.06028075287069421,
    0.02004987990487861,
])

# UV LIP points as in LIP_UV.py.
_ARCF4_UV_DIRECT_X = np.array([100.0, 67.0, 10.0, 5.0])
_ARCF4_UV_DIRECT_Y = np.array([
    0.03942121448304051,
    0.044033601820777875,
    0.08455236611728804,
    0.06737771706391489,
])
_ARCF4_UV_DIRECT_ERR = 0.25 * np.array([0.04, 0.045, 0.085, 0.068])

# Ar--CF4 IR LIP, direct, GEM 0.050 mm, 1 bar.
_ARCF4_IR_DIRECT_POINTS: list[tuple[float, float, float]] = []
for conc, values, errors in [
    (5.0, {696: 0.0007423954814679462, 727: 0.0, 750: 0.030352019422880832, 763: 0.028478502055388058, 772: 0.009542209996882418, 794: 0.0032255060792284705}, {696: 0.00014864734147261504, 727: 0.0, 750: 0.006077282402925308, 763: 0.0057021543440504, 772: 0.0019106045001854228, 794: 0.0006458321953051536}),
    (10.0, {696: 0.00016606742822049213, 727: 0.00023289581724378054, 750: 0.01646440994283698, 763: 0.010520420255841482, 772: 0.00406238907466698, 794: 0.0}, {696: 3.337848055232784e-05, 727: 4.681055514551999e-05, 750: 0.003309240065745319, 763: 0.0021145365269683238, 772: 0.0008165139677163302, 794: 0.0}),
    (67.0, {696: 5.7976067957635574e-05, 727: 0.00013834145604106908, 750: 0.004112455696967686, 763: 0.0009514346999852515, 772: 0.0003514776387269656, 794: 0.00010916911326505648}, {696: 1.1643395769579259e-05, 727: 2.778326265939665e-05, 750: 0.0008259088784642111, 763: 0.00019107764895222087, 772: 7.058763031059062e-05, 794: 2.1924549841632906e-05}),
]:
    y, yerr = _sum_line_dicts(values, errors)
    _ARCF4_IR_DIRECT_POINTS.append((conc, y, yerr))
_ARCF4_IR_DIRECT_X = np.array([p[0] for p in _ARCF4_IR_DIRECT_POINTS])
_ARCF4_IR_DIRECT_Y = np.array([p[1] for p in _ARCF4_IR_DIRECT_POINTS])
_ARCF4_IR_DIRECT_ERR = np.array([p[2] for p in _ARCF4_IR_DIRECT_POINTS])

# Ar--CF4, Florian (indirect), TH-GEM.  These values are intentionally kept in
# Florian's arbitrary relative normalization.  The runner infers one global
# multiplicative scale from the Ar--CF4 TH-GEM 1 bar, 80/20 point and applies it
# to every indirect Florian point, including IR and He--CF4.
_ARCF4_VIS_INDIRECT_1BAR_X = np.array([20.0, 100.0])
_ARCF4_VIS_INDIRECT_1BAR_Y = np.array([0.3542035420581855, 0.09335376])
_ARCF4_VIS_INDIRECT_1BAR_ERR = np.array([0.07607389619126878, 0.02004987990487861])

_ARCF4_VIS_INDIRECT_50MBAR_X = np.array([20.0, 100.0])
_ARCF4_VIS_INDIRECT_50MBAR_Y = np.array([0.1057123306627957, 0.04372129913229157])
_ARCF4_VIS_INDIRECT_50MBAR_ERR = np.array([0.022704169293163653, 0.25 * 0.04372129913229157])

_ARCF4_UV_INDIRECT_1BAR_X = np.array([20.0, 100.0])
_ARCF4_UV_INDIRECT_1BAR_Y = np.array([0.04599727051007351, 0.03942121448304051])
_ARCF4_UV_INDIRECT_1BAR_ERR = np.array([0.010023071441114785, 0.00859011077564899])

# IR Florian, TH-GEM, in the same arbitrary relative normalization.
_ARCF4_IR_INDIRECT_1BAR_X = np.array([20.0])
_ARCF4_IR_INDIRECT_1BAR_Y = np.array([0.0012121110778869518 + 0.0009963113528044523 + 0.010641617291858389 + 0.005277639438703508 + 0.0018895075477094276 + 0.0])
_ARCF4_IR_INDIRECT_1BAR_ERR = np.sqrt(np.array([0.00026032890460820274])**2 + np.array([0.0002139809196170797])**2 + np.array([0.0022855335813600896])**2 + np.array([0.0011334952044080517])**2 + np.array([0.0004058154727878853])**2 + np.array([0.0])**2)

_ARCF4_IR_INDIRECT_50MBAR_X = np.array([20.0])
_ARCF4_IR_INDIRECT_50MBAR_Y = np.array([0.002246119186430594 + 0.0005887938444242574 + 0.057088337541668865 + 0.02364076551779444 + 0.00668799699933377 + 0.004140203178868282])
_ARCF4_IR_INDIRECT_50MBAR_ERR = np.sqrt(np.array([0.00048240607489727035])**2 + np.array([0.0001264571039365711])**2 + np.array([0.012261041623375114])**2 + np.array([0.005077401488711407])**2 + np.array([0.0014364021289985191])**2 + np.array([0.0008892044450984726])**2)

# He--CF4, LIP (direct), GEM 0.050 mm, 1 bar.
_HECF4_VIS_DIRECT_X = np.array([20.0, 40.0, 100.0])
_HECF4_VIS_DIRECT_Y = np.array([0.05981728, 0.0633149, 0.09335376])
_HECF4_VIS_DIRECT_ERR = np.array([0.016221195011451566, 0.023434172057910808, 0.02018412139031937])

_HECF4_UV_DIRECT_X = np.array([20.0, 40.0, 100.0])
_HECF4_UV_DIRECT_Y = np.array([0.0586689111317149, 0.12606696047521188, 0.03942121448304051])
_HECF4_UV_DIRECT_ERR = np.array([0.014241770732051665, 0.04213797898747448, 0.00859011077564899])

# He--CF4, Florian (indirect), TH-GEM, in Florian's same arbitrary relative
# normalization.  It is scaled by the same factor as the Ar--CF4 indirect data.
_HECF4_VIS_INDIRECT_1BAR_X = np.array([100.0, 20.0])

_HECF4_VIS_INDIRECT_1BAR_Y = np.array([0.3542 * 187.0 / 530.0, 0.3542 * 45.0 / 530.0])
_HECF4_VIS_INDIRECT_1BAR_ERR = 0.25 * np.array([0.3542 * 187.0 / 530.0, 0.3542 * 45.0 / 530.0])

_HECF4_VIS_INDIRECT_300MBAR_X = np.array([100.0, 20.0])
_HECF4_VIS_INDIRECT_300MBAR_Y = np.array([0.3542 * 100.0 / 530.0, 0.3542 * 45.0 / 530.0])
_HECF4_VIS_INDIRECT_300MBAR_ERR = 0.25 * np.array([0.3542 * 100.0 / 530.0, 0.3542 * 45.0 / 530.0])

_HECF4_UV_INDIRECT_300MBAR_X = np.array([20.0])
_HECF4_UV_INDIRECT_300MBAR_Y = np.array([0.0709633243946791])
_HECF4_UV_INDIRECT_300MBAR_ERR = np.array([0.015463319066967298])


def _exp_series(
    *,
    curve_id: str,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    label: str,
    direct: bool,
    marker: str = "o",
    scale_anchor: bool = False,
    scale_anchor_x: float | None = None,
) -> ExperimentalSeries:
    return ExperimentalSeries(
        x=x,
        y=y,
        yerr=yerr,
        label=label,
        marker=marker,
        linestyle="none",
        color_from_curve_id=curve_id,
        markerfacecolor="white" if direct else None,
        markeredgewidth=1.55 if direct else 1.35,
        markersize=6.8,
        capsize=3.4,
        scale_group=None if direct else FLORIAN_INDIRECT_SCALE_GROUP,
        scale_anchor=scale_anchor,
        scale_anchor_x=scale_anchor_x,
        scale_anchor_curve_id=curve_id,
    )


def _channel_curve(
    *,
    curve_id: str,
    label: str,
    component: str,
    pressure_bar: float,
    selection: SecondarySelection,
    fit_name: str = "ArCF4_primary",
    band_mode: str = "sys_stat",
    ocw_config: OCWBandConfig | None = None,
) -> BandCurveConfig:
    return BandCurveConfig(
        id=curve_id,
        label=label,
        fit_name=fit_name,
        component=component,
        pressure=pressure_bar,
        x_grid=X_GRID,
        normalization=ARCF4_SECONDARY_NORM_NE,
        selection=selection,
        x_plot_factor=100.0,
        x_axis="concentration",
        show_stat=False,
        show_syst=False,
        show_total=True,
        band_mode=band_mode,
        ocw_config=ocw_config,
    )


def _visible_curve(*, curve_id: str, label: str, pressure_bar: float, selection: SecondarySelection) -> BandCurveConfig:
    return _channel_curve(
        curve_id=curve_id,
        label=label,
        component="vis",
        pressure_bar=pressure_bar,
        selection=selection,
        band_mode=COMPARATION_VISIBLE_BAND_MODE,
        ocw_config=ARCF4_VISIBLE_OCW,
    )


def _uv_curve(*, curve_id: str, label: str, pressure_bar: float, selection: SecondarySelection) -> BandCurveConfig:
    return _channel_curve(
        curve_id=curve_id,
        label=label,
        component="uv",
        pressure_bar=pressure_bar,
        selection=selection,
        band_mode=COMPARATION_UV_BAND_MODE,
        ocw_config=ARCF4_UV_OCW,
    )


def _ir_curve(*, curve_id: str, label: str, pressure_bar: float, selection: SecondarySelection) -> BandCurveConfig:
    return _channel_curve(
        curve_id=curve_id,
        label=label,
        fit_name="ArCF4_IR_primary",
        component="total",
        pressure_bar=pressure_bar,
        selection=selection,
        band_mode=COMPARATION_IR_BAND_MODE,
        ocw_config=ARCF4_IR_OCW,
    )


def _arcf4_visible_curves() -> tuple[BandCurveConfig, ...]:
    return (
        _visible_curve(
            curve_id="comparation_ArCF4_visible_GEM_1bar",
            label="GEM, 1 bar",
            pressure_bar=1.0,
            selection=_paper_concentration_selection("gem_1bar", pressure_bar=1.0, gap_mm=0.050),
        ),
        _visible_curve(
            curve_id="comparation_ArCF4_visible_THGEM_1bar",
            label="TH-GEM, 1 bar",
            pressure_bar=1.0,
            selection=_paper_concentration_selection("thgem_1bar", pressure_bar=1.0, gap_mm=0.570, gap_atol=2e-2),
        ),
        _visible_curve(
            curve_id="comparation_ArCF4_visible_THGEM_50mbar",
            label="TH-GEM, 0.05 bar",
            pressure_bar=0.05,
            selection=_paper_concentration_selection("thgem_50mbar", pressure_bar=0.05, gap_mm=0.570, gap_atol=2e-2),
        ),
    )


def _hecf4_visible_curves() -> tuple[BandCurveConfig, ...]:
    return (
        _visible_curve(
            curve_id="comparation_HeCF4_visible_GEM_1bar",
            label="GEM, 1 bar",
            pressure_bar=1.0,
            selection=_hecf4_concentration_selection("gem_1bar", pressure_bar=1.0, gap_mm=0.050),
        ),
        _visible_curve(
            curve_id="comparation_HeCF4_visible_THGEM_1bar",
            label="TH-GEM, 1 bar",
            pressure_bar=1.0,
            selection=_hecf4_concentration_selection("thgem_1bar", pressure_bar=1.0, gap_mm=0.570, gap_atol=2e-2),
        ),
        _visible_curve(
            curve_id="comparation_HeCF4_visible_THGEM_300mbar",
            label="TH-GEM, 0.3 bar",
            pressure_bar=0.3,
            selection=_hecf4_concentration_selection("thgem_300mbar", pressure_bar=0.3, gap_mm=0.570, gap_atol=2e-2),
        ),
    )


def _uv_gem_curves() -> tuple[BandCurveConfig, ...]:
    return (
        _uv_curve(
            curve_id="comparation_ArCF4_UV_GEM_1bar",
            label=r"Ar--CF$_4$, GEM, 1 bar",
            pressure_bar=1.0,
            selection=_paper_concentration_selection("gem_1bar", pressure_bar=1.0, gap_mm=0.050),
        ),
        _uv_curve(
            curve_id="comparation_HeCF4_UV_GEM_1bar",
            label=r"He--CF$_4$, GEM, 1 bar",
            pressure_bar=1.0,
            selection=_hecf4_concentration_selection("gem_1bar", pressure_bar=1.0, gap_mm=0.050),
        ),
    )


def _arcf4_ir_curves() -> tuple[BandCurveConfig, ...]:
    return (
        _ir_curve(
            curve_id="comparation_ArCF4_IR_total_GEM_1bar",
            label="GEM, 1 bar",
            pressure_bar=1.0,
            selection=_paper_concentration_selection("gem_1bar", pressure_bar=1.0, gap_mm=0.050),
        ),
        _ir_curve(
            curve_id="comparation_ArCF4_IR_total_THGEM_1bar",
            label="TH-GEM, 1 bar",
            pressure_bar=1.0,
            selection=_paper_concentration_selection("thgem_1bar", pressure_bar=1.0, gap_mm=0.570, gap_atol=2e-2),
        ),
        _ir_curve(
            curve_id="comparation_ArCF4_IR_total_THGEM_50mbar",
            label="TH-GEM, 0.05 bar",
            pressure_bar=0.05,
            selection=_paper_concentration_selection("thgem_50mbar", pressure_bar=0.05, gap_mm=0.570, gap_atol=2e-2),
        ),
    )


def multiband_plots() -> list[MultiBandPlotConfig]:
    ar_vis_curves = _arcf4_visible_curves()
    he_vis_curves = _hecf4_visible_curves()
    uv_curves = _uv_gem_curves()
    ir_curves = _arcf4_ir_curves()
    return [
        MultiBandPlotConfig(
            id="comparation_ArCF4_visible_GEM_THGEM",
            title=r"Secondary Ar--CF$_4$ visible emission",
            curves=ar_vis_curves,
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=r"Secondary visible yield [ph/e$^-$]",
            xscale=VISIBLE_XSCALE,
            yscale=VISIBLE_YSCALE,
            xlim=(3.0, 110.0),
           # ylim=(1e-2, 1.5),
            output=OUTDIR / "comparation_ArCF4_visible_GEM_THGEM.pdf",
            legend_ncol=1,
            legend_loc="lower left",
            experimental_series=(
                _exp_series(curve_id=ar_vis_curves[0].id, x=_ARCF4_VIS_DIRECT_X, y=_ARCF4_VIS_DIRECT_Y, yerr=_ARCF4_VIS_DIRECT_ERR, label="GEM Coimbra, 1 bar", direct=True, marker="o"),
                _exp_series(curve_id=ar_vis_curves[1].id, x=_ARCF4_VIS_INDIRECT_1BAR_X, y=_ARCF4_VIS_INDIRECT_1BAR_Y, yerr=_ARCF4_VIS_INDIRECT_1BAR_ERR, label="TH-GEM Florian, 1 bar", direct=False, marker="s", scale_anchor=True, scale_anchor_x=20.0),
                _exp_series(curve_id=ar_vis_curves[2].id, x=_ARCF4_VIS_INDIRECT_50MBAR_X, y=_ARCF4_VIS_INDIRECT_50MBAR_Y, yerr=_ARCF4_VIS_INDIRECT_50MBAR_ERR, label="TH-GEM Florian, 0.05 bar", direct=False, marker="^"),
            ),
        ),
        MultiBandPlotConfig(
            id="comparation_HeCF4_visible_GEM_THGEM",
            title=r"Secondary He--CF$_4$ visible emission",
            curves=he_vis_curves,
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=r"Secondary visible yield [ph/e$^-$]",
            xscale=VISIBLE_XSCALE,
            yscale=VISIBLE_YSCALE,
            xlim=(10.0, 110.0),
           # ylim=(3e-2, 8e-1),
            output=OUTDIR / "comparation_HeCF4_visible_GEM_THGEM.pdf",
            legend_ncol=1,
            legend_loc="upper left",
            legend_fontsize=8.8,
            experimental_series=(
                _exp_series(curve_id=he_vis_curves[0].id, x=_HECF4_VIS_DIRECT_X, y=_HECF4_VIS_DIRECT_Y, yerr=_HECF4_VIS_DIRECT_ERR, label="GEM Coimbra, 1 bar", direct=True, marker="o"),
                _exp_series(curve_id=he_vis_curves[1].id, x=_HECF4_VIS_INDIRECT_1BAR_X, y=_HECF4_VIS_INDIRECT_1BAR_Y, yerr=_HECF4_VIS_INDIRECT_1BAR_ERR, label="TH-GEM Florian, 1 bar", direct=False, marker="s"),
                _exp_series(curve_id=he_vis_curves[2].id, x=_HECF4_VIS_INDIRECT_300MBAR_X, y=_HECF4_VIS_INDIRECT_300MBAR_Y, yerr=_HECF4_VIS_INDIRECT_300MBAR_ERR, label="TH-GEM Florian, 0.3 bar", direct=False, marker="^"),
            ),
        ),
        MultiBandPlotConfig(
            id="comparation_ArCF4_HeCF4_UV_GEM_1bar",
            title=r"Secondary UV emission, GEM, 1 bar",
            curves=uv_curves,
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=r"Secondary UV yield [ph/e$^-$]",
            xscale=VISIBLE_XSCALE,
            yscale=VISIBLE_YSCALE,
            xlim=(4.0, 120.0),
          #  ylim=(5e-3, 0.6),
            output=OUTDIR / "comparation_ArCF4_HeCF4_UV_GEM_1bar.pdf",
            legend_ncol=1,
            legend_loc="lower left",
            experimental_series=(
                _exp_series(curve_id=uv_curves[0].id, x=_ARCF4_UV_DIRECT_X, y=_ARCF4_UV_DIRECT_Y, yerr=_ARCF4_UV_DIRECT_ERR, label=r"Ar--CF$_4$ GEM Coimbra, 1 bar", direct=True, marker="s"),
                _exp_series(curve_id=uv_curves[1].id, x=_HECF4_UV_DIRECT_X, y=_HECF4_UV_DIRECT_Y, yerr=_HECF4_UV_DIRECT_ERR, label=r"He--CF$_4$ GEM Coimbra, 1 bar", direct=True, marker="D"),
            ),
        ),
        MultiBandPlotConfig(
            id="comparation_ArCF4_IR_total_GEM_THGEM",
            title=r"Secondary Ar--CF$_4$ IR peaks",
            curves=ir_curves,
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=r"Secondary IR yield [ph/e$^-$]",
            xscale=VISIBLE_XSCALE,
            yscale="log",
            xlim=(0.8, 100.0),
            ylim=(1.0e-3, 6.0e-1),
            output=OUTDIR / "comparation_ArCF4_IR_total_GEM_THGEM.pdf",
            legend_ncol=1,
            legend_loc="lower left",
            legend_fontsize=9.2,
            hide_ocw_legend=True,
            experimental_series=(
                _exp_series(curve_id=ir_curves[0].id, x=_ARCF4_IR_DIRECT_X, y=_ARCF4_IR_DIRECT_Y, yerr=_ARCF4_IR_DIRECT_ERR, label="GEM Coimbra, 1 bar", direct=True, marker="o"),
                _exp_series(curve_id=ir_curves[1].id, x=_ARCF4_IR_INDIRECT_1BAR_X, y=_ARCF4_IR_INDIRECT_1BAR_Y, yerr=_ARCF4_IR_INDIRECT_1BAR_ERR, label="TH-GEM Florian, 1 bar", direct=False, marker="s"),
                _exp_series(curve_id=ir_curves[2].id, x=_ARCF4_IR_INDIRECT_50MBAR_X, y=_ARCF4_IR_INDIRECT_50MBAR_Y, yerr=_ARCF4_IR_INDIRECT_50MBAR_ERR, label="TH-GEM Florian, 0.05 bar", direct=False, marker="^"),
            ),
        ),
    ]


def metadata_plots() -> list:
    return []


def _num2(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return r"--"
    return rf"\num{{{float(value):.2g}}}"


def _format_pm(value: float, minus: float, plus: float, precision: int = 2) -> str:
    if not np.isfinite(float(minus)) or not np.isfinite(float(plus)):
        return r"--"
    if np.isclose(float(minus), 0.0, rtol=0.0, atol=1e-15) and np.isclose(float(plus), 0.0, rtol=0.0, atol=1e-15):
        return r"--"
    return rf"$^{{+{_num2(plus)}}}_{{-{_num2(minus)}}}$"


def export_comparation_tables(band_outputs: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Path]:
    """Export a small pure-CF4 table for the Ar--CF4 visible comparison.

    The requested critical-concentration table is taken at 100% CF4 for the
    GEM and TH-GEM visible curves.  It includes the central value (OCW optimum
    if available) and the stat./syst./OCW/total asymmetric errors.
    """
    fig_id = "comparation_ArCF4_visible_GEM_THGEM"
    curves = band_outputs.get(fig_id, {})
    wanted = [
        ("GEM, 1 bar", "comparation_ArCF4_visible_GEM_1bar"),
        ("TH-GEM, 1 bar", "comparation_ArCF4_visible_THGEM_1bar"),
    ]
    rows: list[dict[str, object]] = []
    for label, curve_id in wanted:
        df = curves.get(curve_id)
        if df is None or df.empty:
            continue
        xcol = "concentration_percent" if "concentration_percent" in df.columns else ("concentration" if "concentration" in df.columns else "x")
        x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
        target = 100.0 if xcol == "concentration_percent" else 1.0
        idx = int(np.nanargmin(np.abs(x - target)))
        value = float(df["ocw_optimum"].iloc[idx]) if "ocw_optimum" in df.columns else float(df["central"].iloc[idx])
        stat_minus = max(0.0, value - float(df["stat_low"].iloc[idx])) if "stat_low" in df.columns else 0.0
        stat_plus = max(0.0, float(df["stat_high"].iloc[idx]) - value) if "stat_high" in df.columns else 0.0
        syst_minus = max(0.0, value - float(df["syst_low"].iloc[idx])) if "syst_low" in df.columns else 0.0
        syst_plus = max(0.0, float(df["syst_high"].iloc[idx]) - value) if "syst_high" in df.columns else 0.0
        ocw_minus = max(0.0, value - float(df["ocw_low"].iloc[idx])) if "ocw_low" in df.columns else 0.0
        ocw_plus = max(0.0, float(df["ocw_high"].iloc[idx]) - value) if "ocw_high" in df.columns else 0.0
        total_minus = max(0.0, value - float(df["total_low"].iloc[idx])) if "total_low" in df.columns else 0.0
        total_plus = max(0.0, float(df["total_high"].iloc[idx]) - value) if "total_high" in df.columns else 0.0
        rows.append(
            {
                "device": label,
                "concentration_percent": 100.0,
                "yield_ph_per_e": value,
                "stat_minus": stat_minus,
                "stat_plus": stat_plus,
                "syst_minus": syst_minus,
                "syst_plus": syst_plus,
                "ocw_minus": ocw_minus,
                "ocw_plus": ocw_plus,
                "total_minus": total_minus,
                "total_plus": total_plus,
            }
        )

    out: dict[str, Path] = {}
    if not rows:
        return out

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    csv_path = TABLES_DIR / "secondary_comparation_pure_cf4_gain100.csv"
    df_out.to_csv(csv_path, index=False)
    out["csv"] = csv_path

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Secondary visible yield at pure CF$_4$ and gain $\sim 100$ for the Ar--CF$_4$ comparison plots.}",
        r"\label{tab:secondary_comparation_pure_cf4_gain100}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Device & $Y$ [ph/e$^-$] & Stat. & Syst. & OCW & Total \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['device']} & "
            f"{_num2(row['yield_ph_per_e'])} & "
            f"{_format_pm(0.0, row['stat_minus'], row['stat_plus'])} & "
            f"{_format_pm(0.0, row['syst_minus'], row['syst_plus'])} & "
            f"{_format_pm(0.0, row['ocw_minus'], row['ocw_plus'])} & "
            f"{_format_pm(0.0, row['total_minus'], row['total_plus'])} " + r"\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = TABLES_DIR / "secondary_comparation_pure_cf4_gain100.tex"
    tex_path.write_text("\n".join(tex_lines))
    out["tex"] = tex_path
    return out


def config_comparation() -> dict[str, list[MultiBandPlotConfig]]:
    return {"multiband": multiband_plots()}
