#!/usr/bin/env python3
"""Prepare all curated CSV inputs used by the analysis pipeline.

This runner converts the raw experimental pickles, Degrad TXT outputs and
Garfield++ ROOT files into the long/flat CSV tables consumed by the fitting,
prediction and plotting modules.

Typical use from the repository root:

    python data/run_analysis.py

or, from inside data/:

    python run_analysis.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))


@dataclass(frozen=True)
class AnalysisStep:
    key: str
    module_name: str
    description: str

    def load(self) -> ModuleType:
        return importlib.import_module(self.module_name)

    def run(self) -> None:
        module = self.load()
        main: Callable[[], None] | None = getattr(module, "main", None)
        if main is None:
            raise AttributeError(f"{self.module_name} does not expose a main() function")
        main()


STEPS: tuple[AnalysisStep, ...] = (
    AnalysisStep(
        key="experimental",
        module_name="Analysis_experimental",
        description="experimental yield pickles -> data/Experimental/*/csv/*.csv",
    ),
    AnalysisStep(
        key="spectra",
        module_name="Analysis_spectra",
        description="raw spectrum pickles -> data/Spectra/*.csv",
    ),
    AnalysisStep(
        key="primary-degrad",
        module_name="Analysis_primary_degrad",
        description="Degrad TXT files -> data/Primary_DegradData/*.csv",
    ),
    AnalysisStep(
        key="secondary-garfield",
        module_name="Analysis_secondary_garfield",
        description="Garfield++ ROOT files -> data/Secondary_GarfieldData/*/populations/*.csv",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the curated CSV inputs used by fits, predictions and spectra.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[step.key for step in STEPS],
        help="Run only the selected preparation steps.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=[step.key for step in STEPS],
        default=(),
        help="Skip selected preparation steps.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later steps if one preparation step fails.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available preparation steps and exit.",
    )
    return parser.parse_args(argv)


def selected_steps(args: argparse.Namespace) -> list[AnalysisStep]:
    steps = list(STEPS)
    if args.only:
        wanted = set(args.only)
        steps = [step for step in steps if step.key in wanted]
    if args.skip:
        skipped = set(args.skip)
        steps = [step for step in steps if step.key not in skipped]
    return steps


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list:
        for step in STEPS:
            print(f"{step.key:20s} {step.description}")
        return

    print(f"[analysis] repository root: {ROOT_DIR}")
    print(f"[analysis] data directory : {DATA_DIR}")

    failures: list[tuple[str, Exception]] = []
    for step in selected_steps(args):
        print(f"\n[analysis] {step.key}: {step.description}")
        try:
            step.run()
        except Exception as exc:  # noqa: BLE001 - command-line runner should report any failure clearly.
            failures.append((step.key, exc))
            print(f"[analysis:ERROR] {step.key}: {exc}")
            if not args.continue_on_error:
                raise

    if failures:
        failed = ", ".join(key for key, _ in failures)
        raise SystemExit(f"analysis finished with failed steps: {failed}")

    print("\n[analysis] all requested input-preparation steps finished")


if __name__ == "__main__":
    main()
