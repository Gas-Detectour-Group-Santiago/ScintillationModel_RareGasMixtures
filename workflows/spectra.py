#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import sys

from _bootstrap import ROOT


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split("|") if part.strip())


def _is_active(row: dict[str, str]) -> bool:
    return str(row.get("status", "active")).strip().lower() in {"active", "enabled", "true", "1", "yes"}


def _groups(root: Path = ROOT) -> list[str]:
    path = root / "config" / "plots" / "spectra.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if _is_active(row)]
    if not rows:
        raise RuntimeError(f"No active spectra recipes found in {path}")

    groups: list[str] = ["raw"]
    standard: list[str] = []
    extended: list[str] = []
    for row in rows:
        plot_type = str(row.get("plot_type", "")).strip()
        gases = _split(row.get("gas", ""))
        if plot_type in {"generated", "comparison"}:
            standard.extend(gases)
        elif plot_type == "generated_extended":
            extended.extend(gases)
    groups.extend(f"standard:{gas}" for gas in dict.fromkeys(standard))
    groups.extend(f"extended:{gas}" for gas in dict.fromkeys(extended))
    return groups


def _runtime():
    # Imported lazily: importing the scientific stack can retain hundreds of MB.
    # In --exec-group mode the process is immediately replaced by the actual
    # spectrum generator, so no orchestration process remains in memory.
    from scintillation.core.runtime import LegacyRuntime

    return LegacyRuntime.from_root(ROOT)


def _exec_group(group: str) -> None:
    runtime = _runtime()
    runtime.prepare()
    script = runtime.root / "spectra" / "run_all_spectra.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    env = runtime.environment()
    command = [sys.executable, script.name, "--internal-group", group]
    print(f"[workflow] spectra group: {group}", flush=True)
    print(f"[legacy] cwd=spectra :: {' '.join(command)}", flush=True)
    os.chdir(script.parent)
    os.execvpe(sys.executable, command, env)


def _collect() -> None:
    runtime = _runtime()
    runtime.prepare()
    runtime.collect("spectra")
    print("[workflow] spectra complete", flush=True)


def _orchestrate() -> None:
    # Replace this process with a tiny shell. Each group then replaces its own
    # launcher with the legacy spectrum process. This keeps only one scientific
    # Python process in memory at a time, which is essential for the extended
    # million-row spectral grids.
    executable = shlex.quote(sys.executable)
    this_file = shlex.quote(str(Path(__file__).resolve()))
    commands = [
        f"{executable} {this_file} --exec-group {shlex.quote(group)}"
        for group in _groups()
    ]
    commands.append(f"{executable} {this_file} --collect-only")
    os.execvpe("bash", ["bash", "-c", " && ".join(commands)], os.environ.copy())


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CSV-driven spectrum recipes")
    parser.add_argument("--exec-group", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--collect-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--list-groups", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list_groups:
        print("\n".join(_groups()))
    elif args.collect_only:
        _collect()
    elif args.exec_group:
        _exec_group(args.exec_group)
    else:
        _orchestrate()


if __name__ == "__main__":
    cli()
