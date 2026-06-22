from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .auxiliares.model_adapters import PrimaryModelAdapter
from .auxiliares.prediction_types import (
    BandCurveConfig,
    BandPlotConfig,
    MultiBandPlotConfig,
    NormalizationConfig,
    PredictionPoint,
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
from primary_fits.ArN2_fit import CONFIG as ARN2_CONFIG  # noqa: E402
from primary_fits.ArCF4_IR_fit import CONFIG as ARCF4_IR_CONFIG  # noqa: E402
from primary_fits.ArN2_IR_fit import CONFIG as ARN2_IR_CONFIG  # noqa: E402


def _sum_components(*funcs):
    def total(params, degrad, concentration, pressure):
        out = None
        for func in funcs:
            value = np.asarray(func(params, degrad, concentration, pressure), dtype=float)
            out = value if out is None else out + value
        return out

    return total


PRIMARY_ADAPTERS = {
    "ArCF4_primary": PrimaryModelAdapter(
        fit_name="ArCF4_primary",
        degrad_csv=ARCF4_CONFIG.degrad_csv,
        components={
            "uv": ARCF4_CONFIG.equations["uv"],
            "vis": ARCF4_CONFIG.equations["vis"],
        },
    ),
    "ArN2_primary": PrimaryModelAdapter(
        fit_name="ArN2_primary",
        degrad_csv=ARN2_CONFIG.degrad_csv,
        components={
            "uv": ARN2_CONFIG.equations["vis"],
        },
    ),
    "ArCF4_IR_primary": PrimaryModelAdapter(
        fit_name="ArCF4_IR_primary",
        degrad_csv=ARCF4_IR_CONFIG.degrad_csv,
        components={
            "696": ARCF4_IR_CONFIG.equations["696"],
            "727": ARCF4_IR_CONFIG.equations["727"],
            "750": ARCF4_IR_CONFIG.equations["750"],
            "763": ARCF4_IR_CONFIG.equations["763"],
            "772": ARCF4_IR_CONFIG.equations["772"],
            "total": _sum_components(*[ARCF4_IR_CONFIG.equations[k] for k in ("696", "727", "750", "763", "772")]),
        },
    ),
    "ArN2_IR_primary": PrimaryModelAdapter(
        fit_name="ArN2_IR_primary",
        degrad_csv=ARN2_IR_CONFIG.degrad_csv,
        components={
            "696": ARN2_IR_CONFIG.equations["696"],
            "727": ARN2_IR_CONFIG.equations["727"],
            "750": ARN2_IR_CONFIG.equations["750"],
            "763": ARN2_IR_CONFIG.equations["763"],
            "772": ARN2_IR_CONFIG.equations["772"],
            "total": _sum_components(*[ARN2_IR_CONFIG.equations[k] for k in ("696", "727", "750", "763", "772")]),
        },
    ),
}


OWN_NORM = NormalizationConfig(mode="own_norm", output_unit="ph/MeV")
COMMON_ARCF4_NORM = NormalizationConfig(
    mode="reference_norm",
    reference_fit_name="ArCF4_primary",
    output_unit="ph/MeV",
)
ARCF4_PRIMARY_NORM = NormalizationConfig(
    mode="reference_norm",
    reference_fit_name="ArCF4_primary",
    output_unit="ph/MeV",
)
ARN2_PRIMARY_NORM = NormalizationConfig(
    mode="reference_norm",
    reference_fit_name="ArN2_primary",
    output_unit="ph/MeV",
)


def selected_primary_points(normalization: NormalizationConfig) -> list[PredictionPoint]:
    return [
        PredictionPoint(
            id="Ar3rd_UV_Ar",
            label=r"$Y_{\mathrm{Ar3rd,UV}}(100\%\,\mathrm{Ar})$",
            gas="ArCF4",
            channel="uv",
            fit_name="ArCF4_primary",
            component="uv",
            concentration=1e-5,
            pressure=1.0,
            normalization=normalization,
        ),
        PredictionPoint(
            id="CF4_UV_CF4",
            label=r"$Y_{\mathrm{CF_4,UV}}(100\%\,\mathrm{CF_4})$",
            gas="ArCF4",
            channel="uv",
            fit_name="ArCF4_primary",
            component="uv",
            concentration=1.0,
            pressure=1.0,
            normalization=normalization,
        ),
        PredictionPoint(
            id="CF4_VIS_CF4",
            label=r"$Y_{\mathrm{CF_4,VIS}}(100\%\,\mathrm{CF_4})$",
            gas="ArCF4",
            channel="vis",
            fit_name="ArCF4_primary",
            component="vis",
            concentration=1.0,
            pressure=1.0,
            normalization=normalization,
        ),
        PredictionPoint(
            id="N2_UV_N2",
            label=r"$Y_{\mathrm{N_2,UV}}(100\%\,\mathrm{N_2})$",
            gas="ArN2",
            channel="uv",
            fit_name="ArN2_primary",
            component="uv",
            concentration=1.0,
            pressure=1.0,
            normalization=normalization,
        ),
        PredictionPoint(
            id="ArCF4_IR_Ar",
            label=r"$Y_{\mathrm{ArCF_4,IR}}(100\%\,\mathrm{Ar})$",
            gas="ArCF4",
            channel="ir",
            fit_name="ArCF4_IR_primary",
            component="total",
            concentration=1e-5,
            pressure=1.0,
            normalization=ARCF4_PRIMARY_NORM,
        ),
        PredictionPoint(
            id="ArN2_IR_Ar",
            label=r"$Y_{\mathrm{ArN_2,IR}}(100\%\,\mathrm{Ar})$",
            gas="ArN2",
            channel="ir",
            fit_name="ArN2_IR_primary",
            component="total",
            concentration=1e-5,
            pressure=1.0,
            normalization=ARN2_PRIMARY_NORM,
        ),
    ]


def primary_band_plots(normalization: NormalizationConfig = OWN_NORM) -> list[BandPlotConfig]:
    return [
        BandPlotConfig(
            id="ArCF4_primary_uv",
            fit_name="ArCF4_primary",
            component="uv",
            pressure=1.0,
            x_grid=np.logspace(-5, 0, 700),
            normalization=normalization,
            title=r"Primary Ar--CF$_4$ UV prediction",
            xlabel=r"CF$_4$ concentration [\%]",
            ylabel=r"Yield [ph/MeV]",
            xlim=(1e-3, 110),
            output=PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "ArCF4_primary_uv_bands.pdf",
        ),
        BandPlotConfig(
            id="ArCF4_primary_vis",
            fit_name="ArCF4_primary",
            component="vis",
            pressure=1.0,
            x_grid=np.logspace(-5, 0, 700),
            normalization=normalization,
            title=r"Primary Ar--CF$_4$ visible prediction",
            xlabel=r"CF$_4$ concentration [\%]",
            ylabel=r"Yield [ph/MeV]",
            xlim=(8e-2, 110),
            ylim=(6e1, 4e3),
            output=PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "ArCF4_primary_vis_bands.pdf",
        ),
        BandPlotConfig(
            id="ArN2_primary_uv",
            fit_name="ArN2_primary",
            component="uv",
            pressure=1.0,
            x_grid=np.logspace(-4, 0, 700),
            normalization=normalization,
            title=r"Primary Ar--N$_2$ UV prediction",
            xlabel=r"N$_2$ concentration [\%]",
            ylabel=r"Yield [ph/MeV]",
            xlim=(1e-2, 110),
            output=PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "ArN2_primary_uv_bands.pdf",
        ),
        BandPlotConfig(
            id="ArCF4_IR_primary_total",
            fit_name="ArCF4_IR_primary",
            component="total",
            pressure=1.0,
            x_grid=np.logspace(-5, 0, 700),
            normalization=ARCF4_PRIMARY_NORM,
            title=r"Primary Ar--CF$_4$ IR prediction",
            xlabel=r"CF$_4$ concentration [\%]",
            ylabel=r"Yield [ph/MeV]",
            xlim=(1e-2, 20),
            output=PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "ArCF4_IR_primary_total_bands.pdf",
        ),
        BandPlotConfig(
            id="ArN2_IR_primary_total",
            fit_name="ArN2_IR_primary",
            component="total",
            pressure=1.0,
            x_grid=np.logspace(-4, 0, 700),
            normalization=ARN2_PRIMARY_NORM,
            title=r"Primary Ar--N$_2$ IR prediction",
            xlabel=r"N$_2$ concentration [\%]",
            ylabel=r"Yield [ph/MeV]",
            xlim=(1e-2, 20),
            output=PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "ArN2_IR_primary_total_bands.pdf",
        ),
    ]


def primary_ir_low_pressure_band_plots(
    normalization: NormalizationConfig = OWN_NORM,
    pressures_mbar=(0.1, 1.0, 10.0, 50.0, 100.0, 1000.0),
) -> list[BandPlotConfig]:
    """IR primary bands from 0.1 mbar upward.

    The model functions expect pressure in bar. Therefore 0.1 mbar is
    represented as 1e-4 bar.
    """

    configs: list[BandPlotConfig] = []
    for gas, fit_name, xlabel, xmin, grid_min in (
        ("ArCF4", "ArCF4_IR_primary", r"CF$_4$ concentration [\%]", 1e-3, 1e-5),
        ("ArN2", "ArN2_IR_primary", r"N$_2$ concentration [\%]", 1e-2, 1e-4),
    ):
        for pressure_mbar in pressures_mbar:
            pressure_bar = float(pressure_mbar) * 1e-3
            tag = str(pressure_mbar).replace(".", "p")
            norm_for_plot = ARCF4_PRIMARY_NORM if gas == "ArCF4" else ARN2_PRIMARY_NORM
            configs.append(
                BandPlotConfig(
                    id=f"{gas}_IR_primary_total_{tag}mbar",
                    fit_name=fit_name,
                    component="total",
                    pressure=pressure_bar,
                    x_grid=np.logspace(np.log10(grid_min), 0, 700),
                    normalization=norm_for_plot,
                    title=rf"Primary {gas.replace('Ar', 'Ar--')} IR prediction, {pressure_mbar:g} mbar",
                    xlabel=xlabel,
                    ylabel=r"Yield [ph/MeV]",
                    xlim=(xmin, 110),
                    output=PROJECT_ROOT
                    / "primary_predictions"
                    / "plots"
                    / "primary_bands"
                    / "low_pressure_ir"
                    / f"{gas}_IR_primary_total_{tag}mbar_bands.pdf",
                )
            )
    return configs


def _arcf4_ir_curve_for_bar(pressure_bar: float) -> BandCurveConfig:
    tag = str(pressure_bar).replace(".", "p")
    return BandCurveConfig(
        id=f"ArCF4_IR_primary_total_{tag}bar",
        label=f"{pressure_bar:g} bar",
        fit_name="ArCF4_IR_primary",
        component="total",
        pressure=pressure_bar,
        x_grid=np.logspace(-5, 0, 700),
        normalization=ARCF4_PRIMARY_NORM,
        show_stat=False,
        show_syst=False,
        show_total=True,
        paper_overlay_id="ArCF4_IR_primary_total",
    )


def _arcf4_ir_curve_for_mbar(pressure_mbar: float) -> BandCurveConfig:
    pressure_bar = float(pressure_mbar) * 1e-3
    tag = str(pressure_mbar).replace(".", "p")
    return BandCurveConfig(
        id=f"ArCF4_IR_primary_total_{tag}mbar",
        label=f"{pressure_mbar:g} mbar",
        fit_name="ArCF4_IR_primary",
        component="total",
        pressure=pressure_bar,
        x_grid=np.logspace(-5, 0, 700),
        normalization=ARCF4_PRIMARY_NORM,
        show_stat=False,
        show_syst=False,
        show_total=True,
        paper_overlay_id="ArCF4_IR_primary_total",
    )


def arcf4_ir_multiband_plots() -> list[MultiBandPlotConfig]:
    outdir = PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "multibar_ir"
    one_to_five_bar = MultiBandPlotConfig(
        id="ArCF4_IR_primary_total_1to5bar_overlay",
        title=r"Primary Ar--CF$_4$ IR prediction, 1--5 bar",
        curves=[_arcf4_ir_curve_for_bar(p) for p in (1.0, 2.0, 3.0, 4.0, 5.0)],
        xlabel=r"CF$_4$ concentration [\%]",
        ylabel=r"Yield [ph/MeV]",
        xlim=(1e-3, 110),
        ylim=(5e0, 2e3),
        output=outdir / "ArCF4_IR_primary_total_1to5bar_overlay.pdf",
        legend_ncol=2,
        legend_loc="best",
    )
    low_pressure = MultiBandPlotConfig(
        id="ArCF4_IR_primary_total_0p1to1000mbar_overlay",
        title=r"Primary Ar--CF$_4$ IR prediction, 0.1 mbar--1 bar",
        curves=[_arcf4_ir_curve_for_mbar(p) for p in (0.1, 1.0, 10.0, 50.0, 100.0, 1000.0)],
        xlabel=r"CF$_4$ concentration [\%]",
        ylabel=r"Yield [ph/MeV]",
        xlim=(1e-3, 110),
        ylim=(1e2, 8e4),
        output=outdir / "ArCF4_IR_primary_total_0p1to1000mbar_overlay.pdf",
        legend_ncol=2,
        legend_loc="best",
    )
    return [one_to_five_bar, low_pressure]
