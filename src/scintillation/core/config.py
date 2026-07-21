from __future__ import annotations

from pathlib import Path
import tomllib

from .paths import ProjectPaths


def load_project_config(root: str | Path | None = None) -> dict:
    paths = ProjectPaths.from_root(root)
    with (paths.config / "project.toml").open("rb") as handle:
        return tomllib.load(handle)


def default_toys(root: str | Path | None = None) -> int:
    cfg = load_project_config(root)
    return int(cfg.get("runtime", {}).get("default_toys", 100))
