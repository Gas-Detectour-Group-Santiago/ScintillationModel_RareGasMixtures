from __future__ import annotations

from .configs import secondary_metadata_plots, secondary_multiband_plots
from .auxiliares.prediction_types import MetadataPlotConfig, MultiBandPlotConfig


CONFIG_NAME = "config_paper"


def multiband_plots() -> list[MultiBandPlotConfig]:
    """Paper-style secondary yield figures."""
    return secondary_multiband_plots()


def metadata_plots() -> list[MetadataPlotConfig]:
    """Paper-style ne/ni diagnostic figures."""
    return secondary_metadata_plots("paper")


def config_paper() -> dict[str, list[MultiBandPlotConfig] | list[MetadataPlotConfig]]:
    return {
        "multiband": multiband_plots(),
        "metadata": metadata_plots(),
    }
