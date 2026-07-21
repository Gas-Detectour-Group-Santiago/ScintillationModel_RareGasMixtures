#!/usr/bin/env python3
"""Fail when repository files exceed the safe normal-Git size threshold."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

MIB = 1024 * 1024


def tracked_or_staged(root: Path, staged: bool) -> list[Path]:
    if staged:
        command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    else:
        command = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
    return [root / line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-mib", type=float, default=90.0)
    parser.add_argument("--staged", action="store_true", help="Inspect only staged additions/changes")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    limit = int(args.limit_mib * MIB)
    offenders: list[tuple[Path, int]] = []
    for path in tracked_or_staged(root, args.staged):
        if path.is_file():
            size = path.stat().st_size
            if size > limit:
                offenders.append((path.relative_to(root), size))

    if not offenders:
        print(f"OK: no checked file exceeds {args.limit_mib:g} MiB")
        return 0

    print(f"ERROR: files larger than {args.limit_mib:g} MiB:", file=sys.stderr)
    for path, size in sorted(offenders):
        print(f"  {size / MIB:8.1f} MiB  {path}", file=sys.stderr)
    print("Move large campaign ROOT files outside normal Git or configure Git LFS.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
