from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Mapping


def _tree_signature(*roots: Path) -> str:
    """Cheap stamp used to refresh the compatibility workspace after code changes."""
    count = 0
    newest = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Bytecode and editor/OS artefacts are not source. Including them
            # made the compatibility workspace rebuild between independent
            # stages, deleting already generated plots.
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.name in {".DS_Store"} or path.name.endswith("~"):
                continue
            count += 1
            try:
                newest = max(newest, path.stat().st_mtime_ns)
            except OSError:
                pass
    return f"{count}:{newest}"

from .config import default_toys
from .outputs import OutputManager
from .paths import ProjectPaths


def _safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_dir() and not destination.is_symlink(): shutil.rmtree(destination)
        else: destination.unlink()
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _merge_tree(source: Path, destination: Path) -> None:
    if not source.exists(): return
    for path in source.rglob("*"):
        target = destination / path.relative_to(source)
        if path.is_dir(): target.mkdir(parents=True, exist_ok=True)
        else: _safe_symlink(path, target)


@dataclass
class LegacyRuntime:
    paths: ProjectPaths

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "LegacyRuntime":
        return cls(ProjectPaths.from_root(root))

    @property
    def root(self) -> Path: return self.paths.runtime

    def prepare(self, *, refresh: bool = False) -> Path:
        ready = self.root / ".ready"
        signature = _tree_signature(
            self.paths.legacy_source / "project",
            self.paths.legacy_source / "data_scripts",
        )
        if refresh and self.root.exists():
            shutil.rmtree(self.root)
        if ready.exists() and ready.read_text(encoding="utf-8").strip() == signature:
            # Configuration is intentionally not copied: it is the canonical live
            # user-facing registry and must remain editable without rebuilding the
            # runtime workspace.
            _safe_symlink(self.paths.root / "config", self.root / "config")
            return self.root
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        source = self.paths.legacy_source / "project"
        for child in source.iterdir():
            destination = self.root / child.name
            shutil.copytree(child, destination) if child.is_dir() else shutil.copy2(child, destination)
        data_dir = self.root / "data"
        shutil.copytree(self.paths.legacy_source / "data_scripts", data_dir)
        _safe_symlink(self.paths.root / "config", self.root / "config")
        self._build_experimental_view(data_dir / "Experimental")
        _merge_tree(self.paths.raw / "degrad", data_dir / "Primary_DegradData")
        _merge_tree(self.paths.processed / "primary", data_dir / "Primary_DegradData")
        _merge_tree(self.paths.raw / "garfield", data_dir / "Secondary_GarfieldData")
        _merge_tree(self.paths.processed / "secondary", data_dir / "Secondary_GarfieldData")
        _merge_tree(self.paths.reference / "levels", data_dir / "Secondary_GarfieldData" / "levels")
        _merge_tree(self.paths.processed / "spectra", data_dir / "Spectra")

        links = {
            "FitResults": self.paths.fit_cache / "products",
            "Parameters": self.paths.fit_cache / "parameters",
            "Predictions": self.paths.prediction_cache,
            "Tables": self.paths.table_cache,
        }
        for name, target in links.items():
            target.mkdir(parents=True, exist_ok=True); _safe_symlink(target, data_dir / name)
        for name, target in {"Thresholds": self.paths.reference / "thresholds",
                             "annotated_input": self.paths.reference / "annotated_input"}.items():
            target.mkdir(parents=True, exist_ok=True); _safe_symlink(target, data_dir / name)

        static_ar2 = self.paths.reference / "parameters" / "Ar2nd_continium.csv"
        fitted_ar2 = self.paths.fit_cache / "parameters" / "Ar2nd_continium.csv"
        if static_ar2.exists() and not fitted_ar2.exists(): shutil.copy2(static_ar2, fitted_ar2)

        # Generated spectral CSVs are cache, not user-facing outputs.
        spectra_csv = self.root / "spectra" / "csv"
        self.paths.spectrum_cache.mkdir(parents=True, exist_ok=True)
        _safe_symlink(self.paths.spectrum_cache, spectra_csv)

        cs_data = self.paths.reference / "cross_sections"
        if (self.root / "cross_sections").exists():
            _safe_symlink(cs_data, self.root / "cross_sections" / "data")
            (self.root / "cross_sections" / "pdf").mkdir(exist_ok=True)
        ready.write_text(f"{signature}\n", encoding="utf-8")
        return self.root

    def _build_experimental_view(self, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        raw, processed = self.paths.raw / "experimental", self.paths.processed / "experimental"
        mixtures=set()
        if raw.exists(): mixtures.update(p.name for p in raw.iterdir() if p.is_dir())
        if processed.exists(): mixtures.update(p.name for p in processed.iterdir() if p.is_dir())
        for mixture in sorted(mixtures):
            out=destination/mixture; out.mkdir(parents=True,exist_ok=True)
            _merge_tree(raw/mixture,out); (out/"csv").mkdir(exist_ok=True); _merge_tree(processed/mixture,out/"csv")

    def environment(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env=os.environ.copy(); src=str(self.paths.src); runtime=str(self.root)
        env["PYTHONPATH"]=os.pathsep.join([src,runtime,env.get("PYTHONPATH","")]).rstrip(os.pathsep)
        env["SCINTILLATION_ROOT"]=str(self.paths.root)
        toys=str(default_toys(self.paths.root))
        for key in ("PRIMARY_FIT_N_TOYS","PRIMARY_FIT_N_STAT_TOYS","PRIMARY_FIT_N_SYST_TOYS",
                    "JOINT_IR_N_TOYS","JOINT_IR_N_STAT_TOYS","JOINT_IR_N_SYST_TOYS"):
            env.setdefault(key,toys)
        env.setdefault("TFM_PLOT_GRID","0"); env.setdefault("TFM_PLOT_USE_LATEX","0")
        # Keep numerical libraries predictable and memory-bounded during the
        # large spectral grids. Users may override any of these explicitly.
        env.setdefault("MALLOC_ARENA_MAX", "2")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        if extra: env.update({str(k):str(v) for k,v in extra.items()})
        return env

    def run(self, relative_script: str | Path, *, args: Iterable[str]=(), extra_env: Mapping[str,str]|None=None) -> None:
        self.prepare(); script=self.root/relative_script
        if not script.is_file(): raise FileNotFoundError(script)
        command=[sys.executable,script.name,*map(str,args)]
        print(f"[legacy] cwd={script.parent.relative_to(self.root)} :: {' '.join(command)}")
        subprocess.run(command,cwd=script.parent,env=self.environment(extra_env),check=True)

    def collect(self, stage: str) -> None:
        out=OutputManager(self.paths.root)
        figure_mappings={
            "fits":[(self.root/"primary_fits"/"plots",self.paths.figures/"fits"),
                    (self.paths.fit_cache/"products",self.paths.figures/"fits"/"correlations")],
            "primary":[(self.root/"primary_predictions"/"plots",self.paths.figures/"primary")],
            "secondary":[(self.root/"secondary_predictions"/"plots",self.paths.figures/"secondary")],
            "spectra":[(self.root/"spectra"/"plots",self.paths.figures/"spectra")],
            "diagnostics":[(self.root/"integral_comparations"/"plots",self.paths.figures/"diagnostics"/"integrals"),
                           (self.root/"cross_sections"/"pdf",self.paths.figures/"diagnostics"/"cross_sections"),
                           (self.root/"populations_histograms"/"pdf",self.paths.figures/"diagnostics"/"populations"),
                           (self.root/"outputs"/"populations"/"plots",self.paths.figures/"diagnostics"/"populations")],
        }
        for source,destination in figure_mappings.get(stage,[]):
            if source.exists(): out.sync_tree(source,destination,replace=True,suffixes=(".pdf",))
        # User-facing tables are LaTeX only and are classified by scientific
        # family. CSV/NPZ products remain in data/cache.
        if self.paths.table_cache.exists():
            if self.paths.tables.exists():
                shutil.rmtree(self.paths.tables)
            for table in self.paths.table_cache.rglob("*.tex"):
                relative = table.relative_to(self.paths.table_cache)
                text = str(relative).lower()
                name = table.name.lower()
                if "reference" in relative.parts:
                    category = "reference"
                elif "param_secondary" in relative.parts or name.startswith("secondary_") or "secondary" in name:
                    category = "secondary"
                elif "spectrum" in name or "spectra" in name or "vuv" in name or "integral" in name:
                    category = "spectra"
                elif "param_stat_syst" in name or name.startswith("arjoint_ir_primary_param"):
                    category = "fits"
                else:
                    category = "primary"
                destination = self.paths.tables / category / relative.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(table, destination)
        if stage=="diagnostics":
            source = self.root/"integral_comparations"/"tables"
            destination = self.paths.tables/"diagnostics"
            if source.exists(): out.sync_tree(source,destination,replace=False,suffixes=(".tex",))
