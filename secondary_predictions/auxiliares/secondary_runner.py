from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .prediction_runner import PredictionRunner
from .prediction_types import BandPlotConfig, MultiBandPlotConfig


class SecondaryPredictionRunner(PredictionRunner):
    """Compatibility wrapper for older secondary prediction entry points.

    The active implementation now lives in :class:`PredictionRunner` and uses
    ``MultiBandPlotConfig``/``BandCurveConfig``.  This wrapper keeps old
    notebooks that do ``from secondary_predictions.auxiliares import
    SecondaryPredictionRunner`` working while routing the work to the new
    runner.
    """

    def __init__(self, project_root: Path, adapters: Mapping[str, object]):
        super().__init__(
            Path(project_root),
            dict(adapters),
            predictions_subdir=Path("Predictions") / "Secondary",
            log_prefix="[secondary_predictions]",
        )

    def run(
        self,
        configs: Sequence[BandPlotConfig | MultiBandPlotConfig],
        *,
        make_plots: bool = True,
        overwrite: bool = True,
        make_multibands: bool | None = None,
    ):
        configs = list(configs)
        if not configs:
            return {}

        if isinstance(configs[0], MultiBandPlotConfig):
            mb_configs = configs
            if not make_plots:
                mb_configs = [replace(config, output=None) for config in mb_configs]
            return self.run_multi_bands(mb_configs, overwrite=overwrite)

        return self.run_bands(configs, make_plots=make_plots, overwrite=overwrite)
