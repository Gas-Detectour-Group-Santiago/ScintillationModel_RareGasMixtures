from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NormalizationConfig:
    """How model output is converted to the requested absolute scale.

    Primary modes keep the previous behaviour.  For Garfield/secondary
    predictions use ``mode='secondary'``: the evaluated model is multiplied by

        output_scale / Nnorm / NPE / NE

    where ``output_scale`` is the X-ray energy used inside the model, and the
    last denominator can be selected with ``SecondarySelection.normalize_by``
    (usually ``'ne'`` or ``'ni'``).
    """

    mode: str = "own_norm"
    reference_fit_name: str | None = None
    fixed_norm: float | None = None
    output_scale: float = 1000.0
    output_unit: str = "ph/MeV"


@dataclass(frozen=True)
class SecondarySelection:
    """Reusable description of a Garfield/secondary subset.

    The explicit fields cover the usual variables.  ``masks`` is the generic
    interface for future cuts: each key is a CSV column or alias and each value
    can be a scalar exact selection, a ``(min, max)`` range, an operator tuple
    such as ``('>=', 80)`` or a dictionary with ``min/max/eq/atol/in``.
    """

    id: str
    gas: str
    # Optional per-plot Garfield source.  If set, the adapter reads only this
    # reference population table instead of the global/default one.
    reference_dir: Path | None = None
    population_csv: Path | None = None
    population_filename: str = "ArCF4_secondary.csv"
    pressure: float | None = None
    pressure_atol: float = 0.026
    pressure_min: float | None = None
    pressure_max: float | None = None
    gap_mm: float | None = None
    gap_atol: float = 1e-6
    gap_min: float | None = None
    gap_max: float | None = None
    electric_field: float | None = None
    electric_field_atol: float = 1e-8
    electric_field_min: float | None = None
    electric_field_max: float | None = None
    # Short aliases that make configs readable: E, Emin, Emax.
    E: float | None = None
    E_atol: float = 1e-8
    Emin: float | None = None
    Emax: float | None = None
    concentration: float | None = None
    concentration_atol: float = 1e-8
    concentration_min: float | None = None
    concentration_max: float | None = None
    npe: float | None = None
    npe_atol: float = 1e-8
    npe_min: float | None = None
    npe_max: float | None = None
    npe_column: str = "npe"
    ne: float | None = None
    ne_atol: float = 1e-8
    ne_min: float | None = None
    ne_max: float | None = None
    ni: float | None = None
    ni_atol: float = 1e-8
    ni_min: float | None = None
    ni_max: float | None = None
    gain: float | None = None
    gain_atol: float = 1e-8
    gain_min: float | None = None
    gain_max: float | None = None
    gain_column: str = "gain"
    normalize_by: str = "ne"
    masks: Mapping[str, object] = field(default_factory=dict)
    extra_masks: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OCWParameterRule:
    """One ad-hoc optical-cascade/unknown-weight variation rule.

    ``low`` and ``high`` define the two envelope parameter values.
    ``optimum`` defines the preferred OCW line. Each value is computed as

        parameter * factor + additive

    and can optionally be clipped to physical limits.
    """

    name: str
    low_factor: float = 1.0
    high_factor: float = 1.0
    optimum_factor: float | None = None
    low_add: float = 0.0
    high_add: float = 0.0
    optimum_add: float | None = None
    clip_min: float | None = 0.0
    clip_max: float | None = None

    def apply(self, value: float, side: str) -> float:
        value = float(value)
        if side == "low":
            out = value * float(self.low_factor) + float(self.low_add)
        elif side == "high":
            out = value * float(self.high_factor) + float(self.high_add)
        elif side == "optimum":
            factor = 1.0 if self.optimum_factor is None else float(self.optimum_factor)
            add = 0.0 if self.optimum_add is None else float(self.optimum_add)
            out = value * factor + add
        else:
            raise ValueError(f"OCW side desconocido: {side!r}")

        if self.clip_min is not None:
            out = max(float(self.clip_min), out)
        if self.clip_max is not None:
            out = min(float(self.clip_max), out)
        return out


@dataclass(frozen=True)
class OCWBandConfig:
    """Band from hand-defined parameter variations.

    With ``use_corners=False`` the lower envelope is evaluated by applying all
    ``low`` rules together and the upper envelope by applying all ``high``
    rules together. With ``use_corners=True`` all low/high corners are evaluated
    and the plotted envelope is the min/max over those corners.
    """

    id: str = "ocw"
    label: str = "OCW"
    rules: Sequence[OCWParameterRule] = field(default_factory=tuple)
    use_corners: bool = False
    strict: bool = True


@dataclass(frozen=True)
class PredictionPoint:
    id: str
    label: str
    fit_name: str
    component: str
    concentration: float
    pressure: float | None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    selection: SecondarySelection | None = None
    gas: str = ""
    channel: str = ""
    note: str = ""


@dataclass(frozen=True)
class BandPlotConfig:
    id: str
    title: str
    fit_name: str
    component: str
    pressure: float | None
    x_grid: Any
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    selection: SecondarySelection | None = None
    xlabel: str = r"Concentration [\%]"
    ylabel: str = r"Yield [ph/MeV]"
    x_plot_factor: float = 100.0
    x_axis: str = "concentration"
    xscale: str = "log"
    yscale: str = "log"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    output: Path | None = None
    show_stat: bool = True
    show_syst: bool = True
    show_total: bool = True
    band_mode: str = "sys_stat"
    ocw_config: OCWBandConfig | None = None
    overlays: Sequence[str] = field(default_factory=tuple)

    @property
    def grid(self) -> np.ndarray:
        return np.asarray(self.x_grid, dtype=float)


@dataclass(frozen=True)
class BandCurveConfig:
    """One reusable prediction curve for a multi-band overlay plot."""

    id: str
    label: str
    fit_name: str
    component: str
    pressure: float | None
    x_grid: Any
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    selection: SecondarySelection | None = None
    x_plot_factor: float = 100.0
    x_axis: str = "concentration"
    show_stat: bool = False
    show_syst: bool = False
    show_total: bool = True
    band_mode: str = "sys_stat"
    ocw_config: OCWBandConfig | None = None
    paper_overlay_id: str | None = None

    @property
    def grid(self) -> np.ndarray:
        return np.asarray(self.x_grid, dtype=float)

    def as_band_plot_config(
        self,
        *,
        title: str = "",
        xlabel: str = r"Concentration [\%]",
        ylabel: str = r"Yield [ph/MeV]",
        xscale: str = "log",
        yscale: str = "log",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        output: Path | None = None,
    ) -> BandPlotConfig:
        return BandPlotConfig(
            id=self.id,
            title=title,
            fit_name=self.fit_name,
            component=self.component,
            pressure=self.pressure,
            x_grid=self.x_grid,
            normalization=self.normalization,
            selection=self.selection,
            xlabel=xlabel,
            ylabel=ylabel,
            x_plot_factor=self.x_plot_factor,
            x_axis=self.x_axis,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            output=output,
            show_stat=self.show_stat,
            show_syst=self.show_syst,
            show_total=self.show_total,
            band_mode=self.band_mode,
            ocw_config=self.ocw_config,
        )


@dataclass(frozen=True)
class CombinedBandCurveConfig:
    """Curve obtained by combining already-defined BandCurveConfig objects.

    For the usual VIS + IR secondary plot, set ``operation="sum"``. The
    central values are summed point by point. Statistical and systematic
    asymmetric errors are combined in quadrature, which is the appropriate
    default when the child curves come from independent fit products.
    """

    id: str
    label: str
    curves: Sequence[BandCurveConfig]
    operation: str = "sum"
    uncertainty_mode: str = "quadrature"
    x_plot_factor: float | None = None
    x_axis: str = "concentration"
    show_stat: bool = False
    show_syst: bool = False
    show_total: bool = True
    band_mode: str = "sys_stat"
    ocw_config: OCWBandConfig | None = None
    paper_overlay_id: str | None = None

    @property
    def grid(self) -> np.ndarray:
        if not self.curves:
            return np.asarray([], dtype=float)
        return np.asarray(self.curves[0].grid, dtype=float)

    @property
    def component(self) -> str:
        return self.operation






@dataclass(frozen=True)
class MetadataCurveConfig:
    """One selected Garfield metadata curve for diagnostic plots."""

    id: str
    label: str
    selection: SecondarySelection
    marker: str = "o"
    linestyle: str = "-"


@dataclass(frozen=True)
class MetadataPlotConfig:
    """Configuration for plots of Garfield metadata such as ne, ni or (ni-ne)/ni."""

    id: str
    title: str
    curves: Sequence[MetadataCurveConfig]
    x_axis: str = "concentration"
    y: str = "ni_minus_ne_over_ni"
    adapter_name: str = "ArCF4_primary"
    xlabel: str = r"Concentration [\%]"
    ylabel: str = r"$(n_i-n_e)/n_i$"
    xscale: str = "log"
    yscale: str = "linear"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    output: Path | None = None
    legend_loc: str = "best"
    legend_ncol: int = 1
    legend_fontsize: float | None = None
    marker: str = "o"
    linestyle: str = "-"
    linewidth: float = 1.5
    markersize: float = 4.8
    group_duplicate_x: bool = True

@dataclass(frozen=True)
class ExperimentalSeries:
    """One experimental series to overlay on a multi-band figure."""

    x: Sequence[float]
    y: Sequence[float]
    yerr: Sequence[float] | None = None
    label: str | None = None
    marker: str = "o"
    linestyle: str = "none"
    color: str | None = None
    color_from_curve_id: str | None = None
    markerfacecolor: str | None = None
    markeredgecolor: str | None = None
    markeredgewidth: float = 1.2
    markersize: float = 6.0
    capsize: float = 2.5
    alpha: float = 1.0
    # Optional global rescaling for arbitrary-normalized experimental series.
    # All series sharing the same scale_group are multiplied by the same factor.
    # A group factor is inferred from the series marked as scale_anchor=True by
    # matching its point at scale_anchor_x to the corresponding optimized model
    # line: ocw_optimum when available, central otherwise.
    scale_group: str | None = None
    scale_anchor: bool = False
    scale_anchor_x: float | None = None
    scale_anchor_curve_id: str | None = None
    scale_model_column: str = "auto"

@dataclass(frozen=True)
class MultiBandPlotConfig:
    """A single plot containing arbitrary prediction curves."""

    id: str
    title: str
    curves: Sequence[BandCurveConfig | CombinedBandCurveConfig]
    xlabel: str = r"Concentration [\%]"
    ylabel: str = r"Yield [ph/MeV]"
    xscale: str = "log"
    yscale: str = "log"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    output: Path | None = None
    legend_loc: str = "best"
    legend_ncol: int = 2
    legend_fontsize: float | None = None
    hide_ocw_legend: bool = False
    experimental_series: Sequence[ExperimentalSeries] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExperimentalOverlay:
    """External points that can be drawn on top of prediction bands."""

    id: str
    csv_path: Path
    x_col: str = "concentration_percent"
    y_col: str = "yield"
    yerr_col: str | None = "yield_err"
    label: str | None = None
    unit: str = "ph/MeV"
    marker: str = "o"
    color: str | None = None
    conditions: Mapping[str, object] = field(default_factory=dict)
