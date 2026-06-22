from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

_START = Path(globals().get("__file__", Path.cwd())).resolve()
_HERE = _START.parent if _START.is_file() else _START
_ROOT = _HERE.parent if _HERE.name == "spectra_generator" else _HERE

if __package__ in {None, ""}:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from spectra_generator.bootstrap import bootstrap_project
    from spectra_generator.cli_utils import add_no_plots, effective_argv
else:
    from .bootstrap import bootstrap_project
    from .cli_utils import add_no_plots, effective_argv

PROJECT_ROOT = bootstrap_project(_START)

from spectra_generator.run_spectra_raw import main as run_raw  # noqa: E402
from spectra_generator.run_spectra_generated import main as run_generated  # noqa: E402
from spectra_generator.run_spectra_comparation import main as run_comparation  # noqa: E402
from spectra_generator.run_spectra_annotated import main as run_annotated  # noqa: E402


VALID_STAGES = ("raw", "generated", "comparation", "comparison", "annotated", "all")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run spectra-generator stages.")
    parser.add_argument(
        "stage",
        nargs="*",
        choices=VALID_STAGES,
        help="Stage(s) to run. Default: all.",
    )
    add_no_plots(parser)
    args = parser.parse_args(effective_argv(argv))

    stages = list(args.stage) or ["all"]
    if "all" in stages:
        stages = ["raw", "generated", "comparation", "annotated"]

    make_plots = not args.no_plots
    for stage in stages:
        if stage == "raw":
            run_raw(make_plots=make_plots)
        elif stage == "generated":
            run_generated(make_plots=make_plots)
        elif stage in {"comparation", "comparison"}:
            run_comparation(make_plots=make_plots)
        elif stage == "annotated":
            run_annotated(make_plots=make_plots)


def run_data_analysis() -> None:
    root = bootstrap_project(_START)
    runpy.run_path(str(root / "data" / "Analysis_spectra.py"), run_name="__main__")


def main_raw() -> None:
    run_raw(make_plots=True)


def main_generated() -> None:
    run_generated(make_plots=True)


def main_comparation() -> None:
    run_comparation(make_plots=True)


def main_annotated() -> None:
    run_annotated(make_plots=True)


if __name__ == "__main__":
    main()
