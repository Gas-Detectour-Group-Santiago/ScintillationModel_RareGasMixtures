from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secondary_predictions.auxiliares import PredictionRunner  # noqa: E402
from secondary_predictions.configs import SECONDARY_ADAPTERS  # noqa: E402
from secondary_predictions import config_comparation, config_paper  # noqa: E402


def main(
    *,
    make_config_paper: bool = True,
    make_config_comparation: bool = True,
    make_paper_multibands: bool = True,
    make_paper_metadata: bool = True,
    overwrite_paper_bands: bool = True,
    overwrite_paper_metadata: bool = True,
    overwrite_comparation_bands: bool = True,
    # Backward-compatible aliases from the previous runner.
    make_multibands: bool | None = None,
    make_metadata: bool | None = None,
    make_comparation: bool | None = None,
    overwrite_bands: bool | None = None,
    overwrite_metadata: bool | None = None,
    overwrite_comparation: bool | None = None,
):
    if make_multibands is not None:
        make_paper_multibands = make_multibands
    if make_metadata is not None:
        make_paper_metadata = make_metadata
    if make_comparation is not None:
        make_config_comparation = make_comparation
    if overwrite_bands is not None:
        overwrite_paper_bands = overwrite_bands
    if overwrite_metadata is not None:
        overwrite_paper_metadata = overwrite_metadata
    if overwrite_comparation is not None:
        overwrite_comparation_bands = overwrite_comparation

    runner = PredictionRunner(
        PROJECT_ROOT,
        SECONDARY_ADAPTERS,
        predictions_subdir=Path("Predictions") / "Secondary",
        log_prefix="[secondary_predictions]",
    )

    out = {}

    if make_config_paper:
        paper_out = {}
        if make_paper_multibands:
            paper_out["multibands"] = runner.run_multi_bands(
                config_paper.multiband_plots(),
                overwrite=overwrite_paper_bands,
            )
        if make_paper_metadata:
            paper_out["metadata"] = runner.run_metadata_plots(
                config_paper.metadata_plots(),
                overwrite=overwrite_paper_metadata,
            )
        out[config_paper.CONFIG_NAME] = paper_out

    if make_config_comparation:
        comparation_out = runner.run_multi_bands(
            config_comparation.multiband_plots(),
            overwrite=overwrite_comparation_bands,
        )
        if hasattr(config_comparation, "export_comparation_tables"):
            try:
                out_tables = config_comparation.export_comparation_tables(comparation_out)
                if out_tables:
                    print(f"[secondary_predictions] comparation tables: {out_tables}")
            except Exception as exc:
                print(f"[secondary_predictions] aviso: no se pudo exportar la tabla de comparation: {exc}")
        out[config_comparation.CONFIG_NAME] = comparation_out

    return out


if __name__ == "__main__":
    main()
