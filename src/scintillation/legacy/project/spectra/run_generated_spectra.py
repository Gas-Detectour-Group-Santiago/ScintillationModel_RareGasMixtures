from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from spectra.auxiliares import export_vuv_prediction_tables, find_project_root, output_dir, run_generated_mosaics


def main() -> None:
    project_root = find_project_root(Path(__file__))
    run_generated_mosaics(project_root, output_dir(project_root))
    export_vuv_prediction_tables(project_root)


if __name__ == "__main__":
    main()
