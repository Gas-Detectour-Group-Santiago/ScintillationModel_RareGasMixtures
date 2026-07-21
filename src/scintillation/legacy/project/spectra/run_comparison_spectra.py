from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from spectra.auxiliares import (
    build_generated_spectra,
    find_project_root,
    output_dir,
    run_comparison_mosaics,
)


def main() -> None:
    project_root = find_project_root(Path(__file__))
    outdir = output_dir(project_root)
    generated = build_generated_spectra(project_root, outdir)
    run_comparison_mosaics(project_root, outdir, generated)


if __name__ == "__main__":
    main()
