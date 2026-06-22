from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from .spectra_types import AnnotatedScriptConfig


class AnnotatedSpectraRunner:
    def __init__(self, project_root: Path, scripts_dir: Path, output_dir: Path):
        self.project_root = Path(project_root).resolve()
        self.scripts_dir = Path(scripts_dir).resolve()
        self.output_dir = Path(output_dir).resolve()

    def run(
        self,
        configs: list[AnnotatedScriptConfig] | tuple[AnnotatedScriptConfig, ...],
        *,
        selected: set[str] | None = None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        scripts_dir_str = str(self.scripts_dir)
        if scripts_dir_str not in sys.path:
            sys.path.insert(0, scripts_dir_str)

        # Force the local annotated helper modules, even in long-lived notebooks
        # where similarly named modules may already have been imported.
        for module_name in ("spectra_annotate", "spectra_units"):
            sys.modules.pop(module_name, None)

        root_str = str(self.project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        old_env_data = os.environ.get("SPECTRA_DATA_DIR")
        old_env_out = os.environ.get("SPECTRA_OUT_DIR")
        os.environ.setdefault("SPECTRA_DATA_DIR", str(self.project_root / "data"))
        os.environ["SPECTRA_OUT_DIR"] = str(self.output_dir)

        try:
            for config in configs:
                if not config.enabled:
                    continue
                if selected is not None and config.name not in selected:
                    continue
                script = Path(config.script)
                if not script.is_absolute():
                    script = self.scripts_dir / script
                if not script.exists():
                    raise FileNotFoundError(f"No existe el script anotado: {script}")

                print(f"[spectra_annotated] {config.name}")
                try:
                    runpy.run_path(str(script), run_name="__main__")
                except Exception as exc:
                    print(f"[spectra_annotated] AVISO: {config.name} no se pudo generar: {exc}")
                    continue
                if config.output_pdf is not None:
                    print(f"[spectra_annotated] PDF esperado: {config.output_pdf}")
        finally:
            if old_env_data is None:
                os.environ.pop("SPECTRA_DATA_DIR", None)
            else:
                os.environ["SPECTRA_DATA_DIR"] = old_env_data
            if old_env_out is None:
                os.environ.pop("SPECTRA_OUT_DIR", None)
            else:
                os.environ["SPECTRA_OUT_DIR"] = old_env_out
