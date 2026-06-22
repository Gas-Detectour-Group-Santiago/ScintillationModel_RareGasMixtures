from __future__ import annotations

import argparse
import sys
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

bootstrap_project(_START)

from spectra_generator.run_spectra_raw import main as run_raw  # noqa: E402
from spectra_generator.run_spectra_generated import main as run_generated  # noqa: E402
from spectra_generator.run_spectra_comparation import main as run_comparation  # noqa: E402
from spectra_generator.run_spectra_annotated import main as run_annotated  # noqa: E402


def main(make_plots: bool = True) -> None:
    run_raw(make_plots=make_plots)
    run_generated(make_plots=make_plots)
    run_comparation(make_plots=make_plots)
    run_annotated(make_plots=make_plots)


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run all spectra-generator stages.")
    add_no_plots(parser)
    args = parser.parse_args(effective_argv(argv))
    main(make_plots=not args.no_plots)


if __name__ == "__main__":
    cli()
