from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NormalizationConfig:
    """How model output is converted to the requested absolute scale.

    The current primary models return a yield normalized by the X-ray energy
    written in the model files, which is numerically in keV.  Therefore the
    default output scale is 1000, i.e. ph/keV -> ph/MeV.

    Modes
    -----
    as_fit:
        Keep the fitted normalization exactly as it appears in the fit.
    own_norm:
        Divide every evaluated parameter vector by its own first parameter.
        This is equivalent to setting Nnorm=1 for models that are linear in
        the first parameter.
    reference_norm:
        Divide by the central first parameter of another fit, e.g. the Ar-CF4
        normalization used for the TFM comparison table. By default the
        evaluated fit's own Nnorm toy is frozen to its central value so Nnorm
        does not contribute to the propagated uncertainty. Set
        propagate_nnorm=True to recover the old behaviour.
    set_norm_one:
        Set the first parameter of every evaluated vector to one before
        calling the model.
    fixed_norm:
        Divide by the explicit fixed_norm value.
    """

    mode: str = "own_norm"
    reference_fit_name: str | None = None
    fixed_norm: float | None = None
    output_scale: float = 1000.0
    output_unit: str = "ph/MeV"
    propagate_nnorm: bool = False


@dataclass(frozen=True)
class PredictionPoint:
    id: str
    label: str
    fit_name: str
    component: str
    concentration: float
    pressure: float
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    gas: str = ""
    channel: str = ""
    note: str = ""


@dataclass(frozen=True)
class BandPlotConfig:
    id: str
    title: str
    fit_name: str
    component: str
    pressure: float
    x_grid: Any
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    xlabel: str = r"Concentration [\%]"
    ylabel: str = r"Yield [ph/MeV]"
    x_plot_factor: float = 100.0
    xscale: str = "log"
    yscale: str = "log"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    output: Path | None = None
    show_stat: bool = True
    show_syst: bool = True
    show_total: bool = True
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
    pressure: float
    x_grid: Any
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    x_plot_factor: float = 100.0
    show_stat: bool = False
    show_syst: bool = False
    show_total: bool = True
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
            xlabel=xlabel,
            ylabel=ylabel,
            x_plot_factor=self.x_plot_factor,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            output=output,
            show_stat=self.show_stat,
            show_syst=self.show_syst,
            show_total=self.show_total,
        )


@dataclass(frozen=True)
class MultiBandPlotConfig:
    """A single plot containing arbitrary prediction curves."""

    id: str
    title: str
    curves: Sequence[BandCurveConfig]
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


@dataclass(frozen=True)
class SecondarySelection:
    """Reusable description of a Garfield/secondary subset.

    This is not yet used by the primary prediction scripts, but it is the
    contract the secondary predictions will use so that gap/electric-field/npe
    casuistics stay in configs rather than inside plotting code.
    """

    id: str
    gas: str
    pressure: float | None = None
    pressure_atol: float = 0.026
    gap_mm: float | None = None
    gap_atol: float = 1e-6
    electric_field: float | None = None
    electric_field_min: float | None = None
    electric_field_max: float | None = None
    npe: float | None = None
    npe_column: str = "npe"
    ne_min: float | None = None
    ne_max: float | None = None
    gain_min: float | None = None
    gain_max: float | None = None
    gain_column: str = "gain"
    normalize_by: str = "ne"
    extra_masks: Mapping[str, object] = field(default_factory=dict)
