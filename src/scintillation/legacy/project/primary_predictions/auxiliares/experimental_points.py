from __future__ import annotations

from pathlib import Path

import pandas as pd

from .prediction_types import ExperimentalOverlay


def write_overlay_template(path: Path) -> None:
    """Create an empty CSV template for future experimental overlays."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "label",
        "gas",
        "channel",
        "source",
        "pressure_bar",
        "gap_mm",
        "electric_field_kVcm",
        "npe",
        "concentration_percent",
        "yield",
        "yield_err",
        "yield_err_stat",
        "yield_err_syst",
        "unit",
    ]
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def load_overlays(overlays: list[ExperimentalOverlay]) -> dict[str, ExperimentalOverlay]:
    return {overlay.id: overlay for overlay in overlays}

