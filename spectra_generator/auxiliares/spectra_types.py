from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


GasMixture = Literal["ArCF4", "ArN2"]
ComparisonSourceKind = Literal["model", "raw"]
RawComparisonNormalisation = Literal["area_to_generated", "direct_W_Nnorm"]


@dataclass(frozen=True)
class GaussianPeak:
    center_nm: float
    sigma_nm: float
    weight: float = 1.0


@dataclass(frozen=True)
class RawReferenceConfig:
    raw_csv: Path
    gas_mixture: GasMixture
    concentration_percent: float
    pressure_bar: float
    spectrum_columns: Sequence[str] = ("mean_spectrum",)
    label: str = "reference"
    color: str = "magenta"
    alpha: float = 0.35
    fill: bool = True
    linewidth: float = 1.4


@dataclass(frozen=True)
class RawSpectraConfig:
    name: str
    gas_mixture: GasMixture
    input_csv: Path
    output_csv: Path
    output_pdf: Path
    concentrations_percent: Sequence[float] | None = None
    pressures_bar: Sequence[float] | None = None
    spectrum_columns: Sequence[str] = ("mean_spectrum",)
    wavelength_range_nm: tuple[float, float] = (180.0, 850.0)
    title: str | None = None
    reference: RawReferenceConfig | None = None
    mosaic_shape: tuple[int, int] = (3, 3)
    figsize: tuple[float, float] = (18.0, 8.0)
    share_y: bool = True
    common_ylim: bool = False
    show_percent_in_titles: bool = False
    ylabel: str = "Intensity (a.u.)"
    xlabel: str = "Wavelength (nm)"


@dataclass(frozen=True)
class GeneratedSpectraConfig:
    name: str
    gas_mixture: GasMixture
    degrad_csv: Path
    degrad_ir_csv: Path
    parameter_csv: Path
    ir_parameter_csv: Path
    norm_parameter_csv: Path
    output_csv: Path
    output_pdf: Path
    output_summary_pdf: Path
    pressures_bar: Sequence[float]
    concentrations_percent: Sequence[float]
    wavelength_min_nm: float
    wavelength_max_nm: float
    wavelength_points: int
    wavelength_range_nm: tuple[float, float]
    title: str


@dataclass(frozen=True)
class ComparisonConfig:
    name: str
    gas_mixture: GasMixture
    raw_csv: Path
    generated_csv: Path
    norm_parameter_csv: Path
    output_csv: Path
    output_pdf: Path
    pressures_bar: Sequence[float]
    concentrations_percent: Sequence[float]
    spectrum_columns: Sequence[str]
    wavelength_range_nm: tuple[float, float]
    raw_plot_scale: float = 1.0e5
    title: str | None = None


@dataclass(frozen=True)
class ComparisonCurveConfig:
    name: str
    gas_mixture: GasMixture
    kind: ComparisonSourceKind
    pressure_bar: float
    color: str
    label: str
    generated_csv: Path | None = None
    raw_csv: Path | None = None
    norm_parameter_csv: Path | None = None
    spectrum_columns: Sequence[str] = ("mean_spectrum",)
    plot_scale: float = 1.0
    linewidth: float = 1.8
    linestyle: str = "-"
    alpha: float = 0.95
    raw_normalisation: RawComparisonNormalisation = "area_to_generated"
    smooth_window: int = 1


@dataclass(frozen=True)
class ComparisonMosaicConfig:
    name: str
    output_csv: Path
    output_pdf: Path
    concentrations_percent: Sequence[float]
    curves: Sequence[ComparisonCurveConfig]
    wavelength_range_nm: tuple[float, float] = (200.0, 800.0)
    title: str | None = None
    mosaic_shape: tuple[int, int] = (2, 2)
    figsize: tuple[float, float] = (9.4, 6.6)
    ylim: tuple[float, float] | None = None
    ylabel: str = r"ph MeV$^{-1}$ nm$^{-1}$"
    xlabel: str = r"$\lambda$ [nm]"


@dataclass(frozen=True)
class SpectrumResult:
    wavelength_nm: object
    components: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotatedScriptConfig:
    name: str
    script: Path
    output_pdf: Path | None = None
    enabled: bool = True
