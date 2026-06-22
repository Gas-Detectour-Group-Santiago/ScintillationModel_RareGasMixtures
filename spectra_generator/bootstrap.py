from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_MARKERS = (
    ("data", "models", "spectra_generator"),
    ("data", "models", "primary_predictions"),
)


def _looks_like_project_root(path: Path) -> bool:
    return any(all((path / marker).exists() for marker in markers) for markers in PROJECT_MARKERS)


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root independently of the current working directory.

    Search order:
      1. environment variable ``SCINTILLATION_MODEL_ROOT``;
      2. parents of ``start`` if provided;
      3. parents of this file.

    The function intentionally does *not* use ``Path.cwd()``. This makes all
    runners reproducible from notebooks, terminals and batch jobs launched from
    arbitrary folders.
    """
    env_root = os.environ.get("SCINTILLATION_MODEL_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if _looks_like_project_root(root):
            return root
        raise RuntimeError(
            "SCINTILLATION_MODEL_ROOT está definido, pero no parece la raíz del proyecto: "
            f"{root}"
        )

    candidates: list[Path] = []
    if start is not None:
        p = Path(start).expanduser().resolve()
        candidates.extend([p, *p.parents])

    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_project_root(candidate):
            return candidate

    raise RuntimeError(
        "No encuentro la raíz del proyecto. Ejecuta con la ruta absoluta del script, "
        "instala el proyecto con `pip install -e /ruta/al/proyecto`, o define "
        "SCINTILLATION_MODEL_ROOT=/ruta/al/proyecto."
    )


def prepend_once(path: str | Path) -> None:
    path_str = str(Path(path).expanduser().resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def bootstrap_project(start: str | Path | None = None) -> Path:
    """Return project root and expose project-local modules on ``sys.path``."""
    root = find_project_root(start)
    prepend_once(root)
    prepend_once(root / "models")
    prepend_once(root / "data")
    return root
