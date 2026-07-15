from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class Parameter:
    name: str
    tex: str
    x0: float
    lower: float
    upper: float
    fixed: bool = False
    fixed_value: float | None = None
    fixed_error: float | None = None
    unit: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    csv_path: Path
    x_col: str
    pressures: Sequence[float]
    output_concentration_name: str
    w_function: Callable | None = None
    w_input_scale: float = 0.01
    max_concentration_percent: float | None = None
    keep_columns: Sequence[str] | None = None
    preprocess_before_w: Callable | None = None
    preprocess: Callable | None = None


@dataclass(frozen=True)
class SystematicSource:
    name: str
    mode: str = "by_dataset"  # global, by_dataset, by_pressure
    datasets: Sequence[str] | None = None
    pressures: Sequence[float] | None = None
    relative: float | None = None
    absolute: float | None = None


@dataclass(frozen=True)
class ToySpec:
    n_stat: int = 10
    n_syst: int = 10
    seed: int = 12345
    n_jobs: int = 1
    show_progress: bool = True
    fit_error_mode: str = "all"
    use_central_as_x0: bool = True
    stat_mode: str = "pointwise"
    syst_sources: Sequence[SystematicSource] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlotSpec:
    name: str
    dataset_key: str
    theory_key: str
    pressures: Sequence[float]
    concentration_grid: object
    title: str
    xlabel: str
    ylabel: str
    output: Path
    x_col: str
    x_plot_factor: float = 100.0
    min_positive_x: float | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    xscale: str = "log"
    yscale: str = "log"
    cmap: str | None = "viridis"
    darken_factor: float = -0.15
    legend_kwargs: dict | None = None
    label_mode: str = "legend"
    activate_components: bool = False
    line_label_fmt: Sequence[str] | None = None
    show_secondary_yaxis: bool = False
    show_only_fit_points: bool = True


@dataclass(frozen=True)
class FitConfig:
    name: str
    model_name: str
    degrad_csv: Path
    datasets: Sequence[DatasetSpec]
    equations: dict[str, Callable]
    parameters: Sequence[Parameter]
    plots: Sequence[PlotSpec] = field(default_factory=tuple)
    is_infrared: bool = False
    first_point_anchor_weight: float = 1.0
    toy_spec: ToySpec = field(default_factory=ToySpec)
    table_caption: str = ""
    table_label: str = ""
    make_correlation_plot: bool = True

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def parameter_tex(self) -> list[str]:
        return [p.tex for p in self.parameters]

    @property
    def x0(self):
        import numpy as np

        return np.asarray(
            [p.fixed_value if (p.fixed and p.fixed_value is not None) else p.x0 for p in self.parameters],
            dtype=float,
        )

    @property
    def bounds(self):
        import numpy as np

        lower = np.asarray([p.lower for p in self.parameters], dtype=float)
        upper = np.asarray([p.upper for p in self.parameters], dtype=float)
        return lower, upper

    @property
    def fixed_idx(self) -> list[int]:
        return [i for i, p in enumerate(self.parameters) if p.fixed]

    @property
    def fixed_values(self) -> list[float]:
        out = []
        for p in self.parameters:
            if p.fixed:
                out.append(p.x0 if p.fixed_value is None else p.fixed_value)
        return out

    @property
    def fixed_error(self) -> float:
        vals = [p.fixed_error for p in self.parameters if p.fixed and p.fixed_error is not None]
        if not vals:
            return float("nan")
        return float(vals[0])
