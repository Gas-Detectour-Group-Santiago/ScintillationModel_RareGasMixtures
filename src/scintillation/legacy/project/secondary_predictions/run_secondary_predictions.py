from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secondary_predictions.auxiliares import PredictionRunner  # noqa: E402
from secondary_predictions.configs import (  # noqa: E402
    PARAM_SECONDARY_FIT_NAMES,
    SECONDARY_ADAPTERS,
)
from secondary_predictions.auxiliares.tables import export_secondary_parameter_tables  # noqa: E402
from secondary_predictions.auxiliares.ar2nd_population_upgrade import ensure_secondary_ar2nd_populations  # noqa: E402
from secondary_predictions import config_comparation, config_paper  # noqa: E402



def _cleanup_obsolete_vuv_outputs() -> None:
    plot_dir = PROJECT_ROOT / "secondary_predictions" / "plots" / "secondary_vuv"
    band_dir = PROJECT_ROOT / "data" / "Predictions" / "Secondary" / "Bands"
    patterns = (
        "*CF4ionic*",
        "*_Monteiro_reference*",
        "Monteiro_reference_points.csv",
        "*cf4_ionic_vuv*",
        "*_monteiro_ref.csv",
    )
    for base in (plot_dir, band_dir):
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.glob(pattern):
                if path.is_file():
                    path.unlink()

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

    if make_config_paper and make_paper_multibands:
        _cleanup_obsolete_vuv_outputs()

    upgrade_reports = ensure_secondary_ar2nd_populations(PROJECT_ROOT)
    for report in upgrade_reports:
        if report.updated_rows:
            print(
                f"[secondary_predictions] Ar2nd populations actualizadas: "
                f"{report.population_csv} ({report.updated_rows} filas)"
            )
        if report.missing_level_csvs:
            print(
                f"[secondary_predictions] aviso: faltan {len(report.missing_level_csvs)} "
                f"tablas de niveles para {report.population_csv}"
            )

    runner = PredictionRunner(
        PROJECT_ROOT,
        SECONDARY_ADAPTERS,
        predictions_subdir=Path("Predictions") / "Secondary",
        log_prefix="[secondary_predictions]",
    )

    out = {}

    parameter_table_configs = []

    if make_config_paper:
        paper_out = {}
        if make_paper_multibands:
            paper_multiband_configs = config_paper.multiband_plots()
            parameter_table_configs.extend(paper_multiband_configs)
            paper_out["multibands"] = runner.run_multi_bands(
                paper_multiband_configs,
                overwrite=overwrite_paper_bands,
            )
        if make_paper_metadata:
            paper_out["metadata"] = runner.run_metadata_plots(
                config_paper.metadata_plots(),
                overwrite=overwrite_paper_metadata,
            )
        out[config_paper.CONFIG_NAME] = paper_out

    if make_config_comparation:
        comparation_configs = config_comparation.multiband_plots()
        parameter_table_configs.extend(comparation_configs)
        comparation_out = runner.run_multi_bands(
            comparation_configs,
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

    if parameter_table_configs:
        try:
            parameter_tables = export_secondary_parameter_tables(
                PROJECT_ROOT,
                parameter_table_configs,
                extra_fit_names=PARAM_SECONDARY_FIT_NAMES,
            )
            if parameter_tables:
                print(f"[secondary_predictions] parameter secondary tables: {parameter_tables}")
                out["param_secondary"] = parameter_tables
        except Exception as exc:
            print(f"[secondary_predictions] aviso: no se pudieron exportar las tablas param_secondary: {exc}")

    return out


if __name__ == "__main__":
    main()
