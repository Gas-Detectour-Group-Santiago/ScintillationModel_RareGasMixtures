from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from spectra import config as cfg
from spectra.auxiliares import (
    build_generated_spectra,
    find_project_root,
    output_dir,
    run_comparison_mosaics,
    run_generated_mosaics,
    run_annotated_figures,
    run_raw_mosaics,
    export_vuv_prediction_tables,
)


def main(
    *,
    make_raw: bool = cfg.MAKE_RAW_MOSAICS,
    make_generated: bool = cfg.MAKE_GENERATED_MOSAICS,
    make_comparison: bool = cfg.MAKE_COMPARISON_MOSAICS,
    make_annotated: bool = cfg.MAKE_ANNOTATED_FIGURES,
) -> None:
    project_root = find_project_root(Path(__file__))
    outdir = output_dir(project_root)

    generated = {}
    if make_generated or make_comparison:
        generated = build_generated_spectra(project_root, outdir)

    if make_raw:
        run_raw_mosaics(project_root, outdir)

    if make_generated:
        run_generated_mosaics(project_root, outdir, generated)
        export_vuv_prediction_tables(project_root)

    if make_comparison:
        run_comparison_mosaics(project_root, outdir, generated)

    if make_annotated:
        run_annotated_figures(project_root, outdir)


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean spectra generator")
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--no-generated", action="store_true")
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument("--no-annotated", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    main(
        make_raw=not args.no_raw,
        make_generated=not args.no_generated,
        make_comparison=not args.no_comparison,
        make_annotated=not args.no_annotated,
    )


if __name__ == "__main__":
    cli()
