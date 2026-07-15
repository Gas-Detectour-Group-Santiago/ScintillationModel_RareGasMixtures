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



def _ar2nd_parameter_csv() -> Path:
    return PROJECT_ROOT / "data" / "Parameters" / "Ar2nd_continium.csv"


def _read_ar2nd_parameters() -> dict[str, float]:
    from Ar2nd_continium import read_ar2nd_parameters

    return read_ar2nd_parameters(_ar2nd_parameter_csv())


def _ar2nd_degrad_csv(gas_mixture: str) -> Path:
    gas_key = gas_mixture.replace("-", "")
    if gas_key not in {"ArCF4", "ArN2"}:
        raise ValueError(f"Degrad Ar2nd no definido para {gas_mixture!r}")
    return PROJECT_ROOT / "data" / "Primary_DegradData" / f"{gas_key}_Ar2nd.csv"


def _read_ar2nd_degrad(gas_mixture: str):
    import pandas as pd

    path = _ar2nd_degrad_csv(gas_mixture)
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el CSV dedicado al segundo continuo: {path}")
    return pd.read_csv(path)


def _ar2nd_reference_component(kind: str):
    """Pure-Ar second-continuum value from the dedicated Ar2nd Degrad table.

    Do not use the ordinary Ar--CF4 primary CSV here: that table stores only the
    optical Ar_dbleStar component used in the UV fit and misses part of the
    Ar(4s/1s)+upper-state precursor population needed for the Ar second
    continuum.  The dedicated *_Ar2nd.csv files contain Ar(1s4,1s5) + Ar(1s2,1s3) + Ar** consistently with the extended spectra.
    """

    def component(params, degrad, concentration, pressure):
        from Ar2nd_continium import theory_yield_ar2nd_continium

        ar2 = dict(_read_ar2nd_parameters())
        ar2["anchor_Ar2nd_to_pure_argon"] = 0.0
        if kind == "fast":
            ar2["triplet_weight"] = 0.0
        elif kind == "total":
            ar2["triplet_weight"] = 1.0
        else:
            raise ValueError(f"Referencia Ar2nd desconocida: {kind!r}")
        return theory_yield_ar2nd_continium(
            ar2,
            _read_ar2nd_degrad("ArCF4"),
            concentration,
            pressure,
            gas_mixture="ArCF4",
            energy_xray_ev=15.0,
        )

    return component


def _cf4_d_to_x_vuv_component(params, degrad, concentration, pressure):
    """Effective CF4+*(D)->CF4+(X) VUV branch around 155 nm.

    The returned value keeps the same raw convention as the base primary
    models, i.e. it is still proportional to Nnorm and divided by the X-ray
    energy written in the Ar--CF4 model.  The runner then applies the selected
    normalisation and the keV -> MeV factor.
    """
    from ArCF4 import theory_yield_uv

    f = np.atleast_1d(np.asarray(concentration, dtype=float))
    _, y_cf4_uv, _, _ = theory_yield_uv(params, degrad, f, pressure, activate_components=True)
    return float(_read_ar2nd_parameters()["Br_CF4_D_to_X"]) * np.asarray(y_cf4_uv, dtype=float)


def _ar2nd_component(gas_mixture: str, energy_xray_kev: float):
    """Build the Ar second-continuum component for a given primary gas model.

    The Ar second continuum is a pure kinetic extension in this table: it is not
    an optical branch fitted through the primary Nnorm. Therefore it only
    receives the keV -> MeV conversion, without division by any primary
    normalisation.
    """

    def component(params, degrad, concentration, pressure):
        from Ar2nd_continium import theory_yield_ar2nd_continium

        ar2_params = dict(_read_ar2nd_parameters())
        ar2_params["triplet_weight"] = 1.0
        return theory_yield_ar2nd_continium(
            ar2_params,
            _read_ar2nd_degrad(gas_mixture),
            concentration,
            pressure,
            gas_mixture=gas_mixture,
            energy_xray_ev=float(energy_xray_kev),
        )

    return component


PRIMARY_ADAPTERS = {
    "ArCF4_primary": PrimaryModelAdapter(
        fit_name="ArCF4_primary",
        degrad_csv=ARCF4_CONFIG.degrad_csv,
        components={
            "uv": ARCF4_CONFIG.equations["uv"],
            "vis": ARCF4_CONFIG.equations["vis"],
            "uv_vis": _sum_components(ARCF4_CONFIG.equations["uv"], ARCF4_CONFIG.equations["vis"]),
            "cf4_d_to_x_vuv": _cf4_d_to_x_vuv_component,
            "ar2_2nd_continium": _ar2nd_component("ArCF4", 15.0),
            "ar2_2nd_fast_reference": _ar2nd_reference_component("fast"),
            "ar2_2nd_total_reference": _ar2nd_reference_component("total"),
        },
    ),
    "ArN2_primary": PrimaryModelAdapter(
        fit_name="ArN2_primary",
        degrad_csv=ARN2_CONFIG.degrad_csv,
        components={
            "uv": ARN2_CONFIG.equations["vis"],
            "ar2_2nd_continium": _ar2nd_component("ArN2", 12.0),
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
NO_NORM_MEV = NormalizationConfig(mode="as_fit", output_scale=1000.0, output_unit="ph/MeV")

# Default requested behaviour: Nnorm is treated as a nuisance global scale and
# must not inflate the propagated errors.  Set this to True to recover the old
# behaviour where reference/fixed-normalised toys keep their own Nnorm toy.
PROPAGATE_NNORM_IN_REFERENCE_NORM = False

COMMON_ARCF4_NORM = NormalizationConfig(
    mode="reference_norm",
    reference_fit_name="ArCF4_primary",
    output_unit="ph/MeV",
    propagate_nnorm=PROPAGATE_NNORM_IN_REFERENCE_NORM,
)
COMMON_ARN2_NORM = NormalizationConfig(
    mode="reference_norm",
    reference_fit_name="ArN2_primary",
    output_unit="ph/MeV",
    propagate_nnorm=PROPAGATE_NNORM_IN_REFERENCE_NORM,
)
ARCF4_PRIMARY_NORM = NormalizationConfig(
    mode="reference_norm",
    reference_fit_name="ArCF4_primary",
    output_unit="ph/MeV",
    propagate_nnorm=PROPAGATE_NNORM_IN_REFERENCE_NORM,
)
ARN2_PRIMARY_NORM = NormalizationConfig(
    mode="reference_norm",
    reference_fit_name="ArN2_primary",
    output_unit="ph/MeV",
    propagate_nnorm=PROPAGATE_NNORM_IN_REFERENCE_NORM,
)


def selected_primary_points(
    normalization: NormalizationConfig,
    *,
    force_common_normalization: bool = False,
) -> list[PredictionPoint]:
    """Reference primary yields used in compact prediction tables.

    By default the IR rows keep the physically associated primary
    normalization of their gas.  For normalization-comparison tables, set
    ``force_common_normalization=True`` so every row is evaluated with the
    same supplied reference normalization.
    """

    arcf4_ir_norm = normalization if force_common_normalization else ARCF4_PRIMARY_NORM
    arn2_ir_norm = ARCF4_PRIMARY_NORM if not force_common_normalization else normalization

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
            normalization=arcf4_ir_norm,
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
            normalization=arn2_ir_norm,
        ),
        PredictionPoint(
            id="ArCF4_VIS_95_5",
            label=r"$Y_{\mathrm{ArCF_4,VIS}}(95/5)$",
            gas="ArCF4",
            channel="vis",
            fit_name="ArCF4_primary",
            component="vis",
            concentration=0.05,
            pressure=1.0,
            normalization=normalization,
            note="Visible Ar--CF4 95/5 prediction at 1 bar.",
        ),
    ]



def vuv_primary_points(normalization: NormalizationConfig = OWN_NORM) -> list[PredictionPoint]:
    """Pure VUV primary predictions exported without toy uncertainties.

    The Ar second continuum is reported from the current kinetic model, without
    applying any pure-Ar absolute anchor. CF4(D->X) is evaluated
    with the own Ar--CF4 normalisation only; the VUV table does not compare it
    against the Ar--N2 primary normalisation.
    """

    return [
        PredictionPoint(
            id="Ar2nd_total_VUV_Ar_pure",
            label=r"$Y_{\mathrm{Ar2nd,total}}(100\%\,\mathrm{Ar})$",
            gas="Ar",
            channel="vuv",
            fit_name="ArCF4_primary",
            component="ar2_2nd_total_reference",
            concentration=1.0e-5,
            pressure=1.0,
            normalization=NO_NORM_MEV,
            note="Total singlet+triplet Ar second-continuum value at 1 bar, using the dedicated Ar2nd precursor table; this matches the extended spectra at 1 bar.",
        ),
        PredictionPoint(
            id="CF4_D_to_X_VUV_CF4",
            label=r"$Y_{\mathrm{CF_4^+(D\to X)}}(100\%\,\mathrm{CF_4})$",
            gas="ArCF4",
            channel="vuv",
            fit_name="ArCF4_primary",
            component="cf4_d_to_x_vuv",
            concentration=1.0,
            pressure=1.0,
            normalization=normalization,
            note="CF4 ionic VUV branch at 155 nm, Br(D->X) times the fitted CF4 UV ionic channel.",
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
            xlabel=r"CF$_4$ concentration [%]",
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
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=r"Yield [ph/MeV]",
            xlim=(8e-2, 110),
            ylim=(6e1, 4e3),
            yscale="linear",
            output=PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "ArCF4_primary_vis_bands.pdf",
        ),
        BandPlotConfig(
            id="ArN2_primary_uv",
            fit_name="ArN2_primary",
            component="uv",
            pressure=1.0,
            x_grid=np.logspace(-4, 0, 700),
            normalization=ARCF4_PRIMARY_NORM,
            title=r"Primary Ar--N$_2$ UV prediction, Ar--CF$_4$ norm.",
            xlabel=r"N$_2$ concentration [%]",
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
            xlabel=r"CF$_4$ concentration [%]",
            ylabel=r"Yield [ph/MeV]",
            xlim=(1e-3, 110),
            output=PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "ArCF4_IR_primary_total_bands.pdf",
        ),
        BandPlotConfig(
            id="ArN2_IR_primary_total",
            fit_name="ArN2_IR_primary",
            component="total",
            pressure=1.0,
            x_grid=np.logspace(-4, 0, 700),
            normalization=ARCF4_PRIMARY_NORM,
            title=r"Primary Ar--N$_2$ IR prediction, Ar--CF$_4$ norm.",
            xlabel=r"N$_2$ concentration [%]",
            ylabel=r"Yield [ph/MeV]",
            xlim=(1e-2, 20),
            output=PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "ArN2_IR_primary_total_bands.pdf",
        ),
    ]


# Numerical pure-Ar proxies used by both the low-pressure plots and tables.
# Keeping them in one place prevents the tabulated values from being evaluated
# at a concentration different from the visible left edge of each plot.
IR_LOW_PRESSURE_PURE_AR_PROXY = {
    "ArCF4_IR_primary": 1.0e-5,  # 0.001% CF4
    "ArN2_IR_primary": 1.0e-4,   # 0.01% N2
}


def _low_pressure_proxy_concentration(fit_name: str) -> float:
    try:
        return float(IR_LOW_PRESSURE_PURE_AR_PROXY[fit_name])
    except KeyError as exc:
        raise KeyError(f"No hay concentración proxy de Ar puro para {fit_name!r}.") from exc


def primary_ir_low_pressure_band_plots(
    normalization: NormalizationConfig = OWN_NORM,
    pressures_mbar=(0.1, 1.0, 10.0, 50.0, 100.0, 1000.0),
) -> list[BandPlotConfig]:
    """IR primary bands from 0.1 mbar upward.

    The model functions expect pressure in bar. Therefore 0.1 mbar is
    represented as 1e-4 bar.
    """

    configs: list[BandPlotConfig] = []
    for gas, fit_name, xlabel in (
        ("ArCF4", "ArCF4_IR_primary", r"CF$_4$ concentration [%]"),
        ("ArN2", "ArN2_IR_primary", r"N$_2$ concentration [%]"),
    ):
        grid_min = _low_pressure_proxy_concentration(fit_name)
        xmin = 100.0 * grid_min
        for pressure_mbar in pressures_mbar:
            pressure_bar = float(pressure_mbar) * 1e-3
            tag = str(pressure_mbar).replace(".", "p")
            norm_for_plot = ARCF4_PRIMARY_NORM
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
                    show_stat=False,
                    show_syst=False,
                    show_total=False,
                    output=PROJECT_ROOT
                    / "primary_predictions"
                    / "plots"
                    / "primary_bands"
                    / "low_pressure_ir"
                    / f"{gas}_IR_primary_total_{tag}mbar_bands.pdf",
                )
            )
    return configs



def pure_ar_low_pressure_ir_points(
    normalization: NormalizationConfig = OWN_NORM,
    pressures_mbar=(0.1, 1.0, 10.0, 50.0, 100.0, 1000.0),
) -> list[PredictionPoint]:
    """Pure-Ar IR predictions used to audit the low-pressure extrapolation.

    The mixture models are evaluated at a tiny additive fraction rather than
    exactly zero.  This keeps the same numerical convention used elsewhere in
    the primary prediction tables while representing the pure-Ar limit.
    """

    points: list[PredictionPoint] = []
    for fit_tag, fit_name, tex_fit in (
        ("ArCF4", "ArCF4_IR_primary", r"ArCF_4"),
        ("ArN2", "ArN2_IR_primary", r"ArN_2"),
    ):
        concentration_proxy = _low_pressure_proxy_concentration(fit_name)
        for pressure_mbar in pressures_mbar:
            pressure_bar = float(pressure_mbar) * 1e-3
            tag = str(pressure_mbar).replace(".", "p")
            points.append(
                PredictionPoint(
                    id=f"{fit_tag}_IR_pure_Ar_{tag}mbar",
                    label=(
                        rf"$Y^{{\mathrm{{{tex_fit}}}}}_{{\mathrm{{IR}}}}"
                        rf"(100\%\,\mathrm{{Ar}},\,{pressure_mbar:g}\,\mathrm{{mbar}})$"
                    ),
                    gas=fit_tag,
                    channel="ir",
                    fit_name=fit_name,
                    component="total",
                    concentration=concentration_proxy,
                    pressure=pressure_bar,
                    normalization=normalization,
                    note=(
                        "Pure-Ar low-pressure IR proxy evaluated at the same "
                        "minimum additive fraction shown in the corresponding plot."
                    ),
                )
            )
    return points

def _ir_multiband_curve(
    *,
    gas: str,
    fit_name: str,
    pressure: float,
    pressure_unit: str,
    x_grid_min: float,
    normalization: NormalizationConfig,
    id_suffix: str = "",
    paper_overlay_id: str | None = None,
    show_total: bool = True,
) -> BandCurveConfig:
    if pressure_unit == "bar":
        pressure_bar = float(pressure)
        label = f"{pressure:g} bar"
        tag = str(pressure).replace(".", "p") + "bar"
    elif pressure_unit == "mbar":
        pressure_bar = float(pressure) * 1e-3
        label = f"{pressure:g} mbar"
        tag = str(pressure).replace(".", "p") + "mbar"
    else:
        raise ValueError(f"pressure_unit desconocida: {pressure_unit!r}")

    return BandCurveConfig(
        id=f"{gas}_IR_primary_total{id_suffix}_{tag}",
        label=label,
        fit_name=fit_name,
        component="total",
        pressure=pressure_bar,
        x_grid=np.logspace(np.log10(x_grid_min), 0, 700),
        normalization=normalization,
        show_stat=False,
        show_syst=False,
        show_total=show_total,
        paper_overlay_id=paper_overlay_id,
    )


def _arcf4_ir_curve_for_bar(pressure_bar: float) -> BandCurveConfig:
    return _ir_multiband_curve(
        gas="ArCF4",
        fit_name="ArCF4_IR_primary",
        pressure=pressure_bar,
        pressure_unit="bar",
        x_grid_min=1e-5,
        normalization=ARCF4_PRIMARY_NORM,
        paper_overlay_id="ArCF4_IR_primary_total",
    )


def _arcf4_ir_curve_for_mbar(pressure_mbar: float) -> BandCurveConfig:
    return _ir_multiband_curve(
        gas="ArCF4",
        fit_name="ArCF4_IR_primary",
        pressure=pressure_mbar,
        pressure_unit="mbar",
        x_grid_min=_low_pressure_proxy_concentration("ArCF4_IR_primary"),
        normalization=ARCF4_PRIMARY_NORM,
        paper_overlay_id="ArCF4_IR_primary_total",
        show_total=False,
    )


def _arn2_ir_curve_for_bar_arcf4_norm(pressure_bar: float) -> BandCurveConfig:
    return _ir_multiband_curve(
        gas="ArN2",
        fit_name="ArN2_IR_primary",
        pressure=pressure_bar,
        pressure_unit="bar",
        x_grid_min=1e-4,
        normalization=ARCF4_PRIMARY_NORM,
        id_suffix="_arcf4norm",
        paper_overlay_id="ArN2_IR_primary_total",
    )


def _arn2_ir_curve_for_mbar_arcf4_norm(pressure_mbar: float) -> BandCurveConfig:
    return _ir_multiband_curve(
        gas="ArN2",
        fit_name="ArN2_IR_primary",
        pressure=pressure_mbar,
        pressure_unit="mbar",
        x_grid_min=_low_pressure_proxy_concentration("ArN2_IR_primary"),
        normalization=ARCF4_PRIMARY_NORM,
        id_suffix="_arcf4norm",
        paper_overlay_id="ArN2_IR_primary_total",
        show_total=False,
    )


def arcf4_ir_multiband_plots() -> list[MultiBandPlotConfig]:
    outdir = PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "multibar_ir"
    one_to_five_bar = MultiBandPlotConfig(
        id="ArCF4_IR_primary_total_1to5bar_overlay",
        title=r"Primary Ar--CF$_4$ IR prediction, 1--3 bar",
        curves=[_arcf4_ir_curve_for_bar(p) for p in (1.0, 2.0, 3.0)],
        xlabel=r"CF$_4$ concentration [%]",
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
        xlabel=r"CF$_4$ concentration [%]",
        ylabel=r"Yield [ph/MeV]",
        xlim=(1e-3, 110),
        ylim=(5e2, 3e4),
        output=outdir / "ArCF4_IR_primary_total_0p1to1000mbar_overlay.pdf",
        legend_ncol=3,
        legend_loc="upper right",
        legend_fontsize=10.0,
    )
    return [one_to_five_bar, low_pressure]


def arn2_ir_multiband_plots_arcf4_norm() -> list[MultiBandPlotConfig]:
    """Ar--N2 IR multibar overlays evaluated with the Ar--CF4 primary scale.

    These plots are intentionally separated from the default Ar--N2 IR bands,
    which use the Ar--N2 primary normalization.  The suffix ``arcf4norm`` avoids
    overwriting the standard Ar--N2 CSV bands.
    """

    outdir = PROJECT_ROOT / "primary_predictions" / "plots" / "primary_bands" / "multibar_ir"
    one_to_five_bar = MultiBandPlotConfig(
        id="ArN2_IR_primary_total_arcf4norm_1to5bar_overlay",
        title=r"Primary Ar--N$_2$ IR prediction, 1--3 bar, Ar--CF$_4$ norm.",
        curves=[_arn2_ir_curve_for_bar_arcf4_norm(p) for p in (1.0, 2.0, 3.0)],
        xlabel=r"N$_2$ concentration [%]",
        ylabel=r"Yield [ph/MeV]",
        xlim=(1e-2, 110),
        ylim=(5e0, 2e3),
        output=outdir / "ArN2_IR_primary_total_arcf4norm_1to5bar_overlay.pdf",
        legend_ncol=2,
        legend_loc="best",
    )
    low_pressure = MultiBandPlotConfig(
        id="ArN2_IR_primary_total_arcf4norm_0p1to1000mbar_overlay",
        title=r"Primary Ar--N$_2$ IR prediction, 0.1 mbar--1 bar, Ar--CF$_4$ norm.",
        curves=[_arn2_ir_curve_for_mbar_arcf4_norm(p) for p in (0.1, 1.0, 10.0, 50.0, 100.0, 1000.0)],
        xlabel=r"N$_2$ concentration [%]",
        ylabel=r"Yield [ph/MeV]",
        xlim=(1e-2, 110),
        ylim=(5e2, 3e4),
        output=outdir / "ArN2_IR_primary_total_arcf4norm_0p1to1000mbar_overlay.pdf",
        legend_ncol=3,
        legend_loc="upper right",
        legend_fontsize=10.0,
    )
    return [one_to_five_bar, low_pressure]
