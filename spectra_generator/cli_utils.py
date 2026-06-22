from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def running_inside_ipykernel() -> bool:
    """Return True when the current process is a Jupyter/IPython kernel."""
    argv0 = sys.argv[0].lower() if sys.argv else ""
    return (
        "ipykernel" in sys.modules
        or "ipykernel_launcher" in argv0
        or any("kernel-" in str(arg) and str(arg).endswith(".json") for arg in sys.argv[1:])
    )


def effective_argv(argv: Sequence[str] | None = None) -> list[str]:
    """Return command-line arguments safe for terminal scripts and notebooks.

    In a normal terminal, ``argv=None`` means ``sys.argv[1:]``. Inside a
    notebook, ``sys.argv`` contains kernel arguments such as ``-f kernel.json``;
    those must not be parsed by our CLIs. Explicit ``argv`` is always respected.
    """
    if argv is not None:
        return list(argv)
    if running_inside_ipykernel():
        return []
    return sys.argv[1:]


def add_no_plots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-plots", action="store_true", help="Write CSVs without PDFs where applicable.")
