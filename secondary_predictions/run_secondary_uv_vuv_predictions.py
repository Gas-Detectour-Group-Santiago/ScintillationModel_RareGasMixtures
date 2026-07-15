from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secondary_predictions import config_paper  # noqa: E402
from secondary_predictions.auxiliares import PredictionRunner  # noqa: E402
from secondary_predictions.auxiliares.ar2nd_population_upgrade import (  # noqa: E402
    ensure_secondary_ar2nd_populations,
)
from secondary_predictions.configs import SECONDARY_ADAPTERS  # noqa: E402


def _selected_configs(*, make_uv: bool, make_vuv: bool):
    selected = []
    for config in config_paper.multiband_plots():
        is_uv = "200_400nm" in config.id
        is_vuv = "100_200nm_Ar2nd" in config.id
        if (make_uv and is_uv) or (make_vuv and is_vuv):
            selected.append(config)
    return selected



def _cleanup_obsolete_vuv_outputs() -> None:
    """Remove outputs from superseded CF4+* and Monteiro VUV variants."""
    plot_dir = PROJECT_ROOT / "secondary_predictions" / "plots" / "secondary_vuv"
    band_dir = PROJECT_ROOT / "data" / "Predictions" / "Secondary" / "Bands"
    patterns = (
        "*CF4ionic*",
        "*_Monteiro_reference*",
        "Monteiro_reference_points.csv",
        "*cf4_ionic_vuv*",
        "*_monteiro_ref.csv",
    )
    removed = 0
    for base in (plot_dir, band_dir):
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.glob(pattern):
                if path.is_file():
                    path.unlink()
                    removed += 1
    if removed:
        print(f"[secondary_predictions] removed {removed} obsolete VUV outputs")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate only the secondary UV (200-400 nm) and/or VUV (100-200 nm) figures."
    )
    parser.add_argument("--uv-only", action="store_true", help="Generate only the 200-400 nm figures.")
    parser.add_argument("--vuv-only", action="store_true", help="Generate only the 100-200 nm figures.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Reuse existing band CSVs/PDFs when possible.",
    )
    args = parser.parse_args()
    if args.uv_only and args.vuv_only:
        parser.error("--uv-only and --vuv-only cannot be used together")

    make_uv = not args.vuv_only
    make_vuv = not args.uv_only
    if make_vuv:
        _cleanup_obsolete_vuv_outputs()

    reports = ensure_secondary_ar2nd_populations(PROJECT_ROOT)
    for report in reports:
        if report.updated_rows:
            print(
                "[secondary_predictions] Ar2nd populations updated: "
                f"{report.population_csv} ({report.updated_rows} rows)"
            )
        if report.missing_level_csvs:
            print(
                "[secondary_predictions] warning: "
                f"{len(report.missing_level_csvs)} level CSV files are missing for "
                f"{report.population_csv}"
            )

    configs = _selected_configs(make_uv=make_uv, make_vuv=make_vuv)
    if not configs:
        raise RuntimeError("No UV/VUV configurations were found.")

    runner = PredictionRunner(
        PROJECT_ROOT,
        SECONDARY_ADAPTERS,
        predictions_subdir=Path("Predictions") / "Secondary",
        log_prefix="[secondary_predictions]",
    )
    outputs = runner.run_multi_bands(configs, overwrite=not args.no_overwrite)
    print(f"[secondary_predictions] generated {len(outputs)} UV/VUV plots")
    for config in configs:
        print(f"[secondary_predictions] plot: {config.output}")


if __name__ == "__main__":
    main()
