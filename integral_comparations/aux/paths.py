from __future__ import annotations

from pathlib import Path


ROOT_MARKERS: tuple[tuple[str, ...], ...] = (
    ("data", "integral_comparations"),
    ("data", "models"),
)


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the project root by walking upwards.

    The integral-comparison scripts only need ``data/`` plus the
    ``integral_comparations/`` package.  Do not require ``models/`` here,
    because lightweight analysis zips often contain the exported spectra but
    not the model package.
    """
    if start is None:
        start_path = Path(__file__).resolve()
    else:
        start_path = Path(start).resolve()

    first = start_path if start_path.is_dir() else start_path.parent
    candidates = [first, *first.parents]

    for candidate in candidates:
        for markers in ROOT_MARKERS:
            if all((candidate / marker).exists() for marker in markers):
                return candidate

    # Normal layout: project/integral_comparations/aux/paths.py
    return Path(__file__).resolve().parents[2]
