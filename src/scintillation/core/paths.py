from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    path = Path(start or __file__).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "config" / "project.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate ScintillationModel root from {start!r}")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        return cls(find_project_root(root))

    @property
    def config(self) -> Path: return self.root / "config"
    @property
    def data(self) -> Path: return self.root / "data"
    @property
    def raw(self) -> Path: return self.data / "raw"
    @property
    def processed(self) -> Path: return self.data / "processed"
    @property
    def reference(self) -> Path: return self.data / "reference"
    @property
    def cache(self) -> Path: return self.data / "cache"
    @property
    def fit_cache(self) -> Path: return self.cache / "fits"
    @property
    def prediction_cache(self) -> Path: return self.cache / "predictions"
    @property
    def spectrum_cache(self) -> Path: return self.cache / "spectra"
    @property
    def secondary_cache(self) -> Path: return self.cache / "secondary"
    @property
    def table_cache(self) -> Path: return self.cache / "tables"
    @property
    def src(self) -> Path: return self.root / "src"
    @property
    def legacy_source(self) -> Path: return self.src / "scintillation" / "legacy"
    @property
    def runtime(self) -> Path: return self.root / ".runtime"
    @property
    def outputs(self) -> Path: return self.root / "outputs"
    @property
    def figures(self) -> Path: return self.outputs / "figures"
    @property
    def tables(self) -> Path: return self.outputs / "tables"
    @property
    def archive(self) -> Path: return self.outputs / "archive"
    @property
    def current(self) -> Path: return self.outputs

    def ensure(self) -> None:
        for path in (self.raw, self.processed, self.reference, self.cache, self.fit_cache,
                     self.prediction_cache, self.spectrum_cache, self.secondary_cache,
                     self.table_cache, self.figures, self.tables, self.archive):
            path.mkdir(parents=True, exist_ok=True)
