from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .fit_exports import (
    build_parameter_summary,
    export_matrix,
    export_stat_syst_latex,
    export_toys,
    export_vector,
)
from .fit_io import ensure_project_paths, load_dataset_triplet, project_root_from_file, write_json
from .fit_toys import make_stat_toy, make_syst_toy, summarize_toys
from .fit_types import FitConfig
from .fiting import fitParameters
from .ploting import plot_fit_vs_experiment_by_pressure


class PrimaryFitRunner:
    def __init__(self, config: FitConfig, project_root: Path | None = None):
        self.config = config
        self.project_root = Path(project_root).resolve() if project_root else project_root_from_file(__file__)
        ensure_project_paths(self.project_root)

        self.data_dir = self.project_root / "data"
        self.fit_results_dir = self.data_dir / "FitResults"
        self.parameters_dir = self.data_dir / "Parameters"
        self.tables_dir = self.data_dir / "Tables"
        self.plot_dir = self.project_root / "primary_fits" / "plots" / "plot_fit"

    @staticmethod
    def _progress(iterable, *, total: int, desc: str, enabled: bool = True):
        if not enabled:
            return iterable

        try:
            from tqdm.auto import tqdm

            return tqdm(
                iterable,
                total=total,
                desc=desc,
                unit="fit",
                leave=True,
                dynamic_ncols=True,
            )
        except ModuleNotFoundError:
            return iterable

    @staticmethod
    def _resolve_n_jobs(n_jobs: int) -> int:
        if n_jobs == 0:
            return 1
        if n_jobs < 0:
            cpu_count = os.cpu_count() or 1
            return max(1, cpu_count + 1 + n_jobs)
        return max(1, int(n_jobs))

    def load_data(self):
        degrad = pd.read_csv(self.config.degrad_csv)
        dataset_triplets = {
            spec.key: load_dataset_triplet(self.project_root, spec)
            for spec in self.config.datasets
        }

        nominal = {k: v["all"] for k, v in dataset_triplets.items()}
        stat_errors = {k: v["stat"] for k, v in dataset_triplets.items()}
        syst_errors = {k: v["syst"] for k, v in dataset_triplets.items()}
        return degrad, nominal, stat_errors, syst_errors

    def fit(self, degrad, experimental_data, x0=None, verbose=0):
        x0_use = self.config.x0 if x0 is None else np.asarray(x0, dtype=float)
        fixed_idx = self.config.fixed_idx
        fixed_values = self.config.fixed_values if fixed_idx else None

        return fitParameters(
            self.config.equations,
            experimental_data,
            degrad,
            x0=x0_use,
            bounds=self.config.bounds,
            is_infrared=self.config.is_infrared,
            fixed_idx=fixed_idx,
            fixed_values=fixed_values,
            fixed_error=self.config.fixed_error,
            verbose=verbose,
        )

    def _run_toy_loop(self, kind, degrad, nominal, stat_errors, syst_errors, central_x):
        spec = self.config.toy_spec
        n_toys = spec.n_stat if kind == "stat" else spec.n_syst
        if n_toys <= 0:
            return np.empty((0, len(central_x)), dtype=float)

        rng = np.random.default_rng(spec.seed + (0 if kind == "stat" else 1000003))
        dataset_specs = list(self.config.datasets)
        x0_toy = central_x if spec.use_central_as_x0 else self.config.x0

        def run_one(seed):
            local_rng = np.random.default_rng(seed)
            if kind == "stat":
                toy_data = make_stat_toy(nominal, stat_errors, dataset_specs, local_rng)
            else:
                toy_data = make_syst_toy(
                    nominal,
                    syst_errors,
                    dataset_specs,
                    local_rng,
                    list(spec.syst_sources),
                )
            try:
                result = self.fit(degrad, toy_data, x0=x0_toy, verbose=0)
                return np.asarray(result.x, dtype=float)
            except Exception:
                return np.full(len(central_x), np.nan, dtype=float)

        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_toys)

        desc = f"{self.config.name} toys {kind}"
        show_progress = bool(getattr(spec, "show_progress", True))
        n_jobs = self._resolve_n_jobs(spec.n_jobs)

        if n_jobs == 1:
            rows = [
                run_one(int(seed))
                for seed in self._progress(seeds, total=n_toys, desc=desc, enabled=show_progress)
            ]
        else:
            rows = []
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(run_one, int(seed)) for seed in seeds]
                for future in self._progress(
                    as_completed(futures),
                    total=n_toys,
                    desc=f"{desc} ({n_jobs} threads)",
                    enabled=show_progress,
                ):
                    rows.append(future.result())

        return np.vstack(rows)

    def run_toys(self, degrad, nominal, stat_errors, syst_errors, central):
        t0 = perf_counter()
        stat = self._run_toy_loop("stat", degrad, nominal, stat_errors, syst_errors, central.x)
        syst = self._run_toy_loop("syst", degrad, nominal, stat_errors, syst_errors, central.x)
        print(
            f"[{self.config.name}] toys: stat={len(stat)}, syst={len(syst)} "
            f"en {perf_counter() - t0:.1f} s"
        )
        return stat, syst

    def export_results(self, central, stat_toys, syst_toys):
        names = self.config.parameter_names
        tex = self.config.parameter_tex

        stat_minus, stat_plus = summarize_toys(central.x, stat_toys)
        syst_minus, syst_plus = summarize_toys(central.x, syst_toys)

        summary = build_parameter_summary(
            names=names,
            tex_names=tex,
            central=central.x,
            central_err=np.asarray(getattr(central, "perr", np.full_like(central.x, np.nan))),
            stat_minus=stat_minus,
            stat_plus=stat_plus,
            syst_minus=syst_minus,
            syst_plus=syst_plus,
            fixed_idx=list(getattr(central, "fixed_idx", [])),
        )

        summary_path = self.parameters_dir / f"{self.config.name}.csv"
        summary.to_csv(summary_path, index=False)

        fit_prefix = self.fit_results_dir / self.config.name
        export_vector(fit_prefix.with_name(f"{self.config.name}_central.csv"), names, central.x)
        export_toys(fit_prefix.with_name(f"{self.config.name}_toys_stat.csv"), names, stat_toys)
        export_toys(fit_prefix.with_name(f"{self.config.name}_toys_syst.csv"), names, syst_toys)

        if hasattr(central, "pcov"):
            export_matrix(fit_prefix.with_name(f"{self.config.name}_covariance.csv"), names, central.pcov)
            diag = np.sqrt(np.clip(np.diag(central.pcov), 0, None))
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = central.pcov / np.outer(diag, diag)
            corr = np.clip(corr, -1, 1)
            export_matrix(fit_prefix.with_name(f"{self.config.name}_correlation.csv"), names, corr)

        write_json(
            fit_prefix.with_name(f"{self.config.name}_metadata.json"),
            {
                "name": self.config.name,
                "model_name": self.config.model_name,
                "chi2": float(getattr(central, "chi2", np.nan)),
                "dof": int(getattr(central, "dof", -1)),
                "chi2_red": float(getattr(central, "chi2_red", np.nan)),
                "success": bool(getattr(central, "success", False)),
                "message": str(getattr(central, "message", "")),
                "n_stat_toys": int(len(stat_toys)),
                "n_syst_toys": int(len(syst_toys)),
            },
        )

        export_stat_syst_latex(
            summary=summary,
            path=self.tables_dir / f"{self.config.name}_param_stat_syst.tex",
            caption=self.config.table_caption or f"Parámetros del ajuste {self.config.name}.",
            label=self.config.table_label or f"tab:{self.config.name}_params",
        )

        return summary

    def plot_fits(self, degrad, nominal, central):
        for plot in self.config.plots:
            output = plot.output
            if not output.is_absolute():
                output = self.project_root / output
            output.parent.mkdir(parents=True, exist_ok=True)

            plot_fit_vs_experiment_by_pressure(
                df_exp=nominal[plot.dataset_key],
                theory_func=self.config.equations[plot.theory_key],
                fit_params=central.x,
                degrad_data=degrad,
                concentration_grid=plot.concentration_grid,
                pressures=list(plot.pressures),
                x_col=plot.x_col,
                x_plot_factor=plot.x_plot_factor,
                min_positive_x=plot.min_positive_x,
                title=plot.title,
                xlabel=plot.xlabel,
                ylabel=plot.ylabel,
                xlim=plot.xlim,
                ylim=plot.ylim,
                xscale=plot.xscale,
                yscale=plot.yscale,
                cmap=plot.cmap,
                darken_factor=plot.darken_factor,
                legend_kwargs=plot.legend_kwargs,
                label_mode=plot.label_mode,
                output=str(output),
                show=False,
                activate_components=plot.activate_components,
                line_label_fmt=plot.line_label_fmt,
                show_secondary_yaxis=plot.show_secondary_yaxis,
                show_only_fit_points=plot.show_only_fit_points,
            )

    def run_all(self, run_toys=True, make_plots=True):
        degrad, nominal, stat_errors, syst_errors = self.load_data()

        print(f"[{self.config.name}] ajuste central...")
        central = self.fit(degrad, nominal)

        print(
            f"[{self.config.name}] chi2={getattr(central, 'chi2', np.nan):.4g}, "
            f"dof={getattr(central, 'dof', np.nan)}, "
            f"chi2_red={getattr(central, 'chi2_red', np.nan):.4g}"
        )
        print(f"[{self.config.name}] x = {central.x}")

        if run_toys:
            stat_toys, syst_toys = self.run_toys(degrad, nominal, stat_errors, syst_errors, central)
        else:
            stat_toys = np.empty((0, len(central.x)), dtype=float)
            syst_toys = np.empty((0, len(central.x)), dtype=float)

        summary = self.export_results(central, stat_toys, syst_toys)

        if make_plots:
            self.plot_fits(degrad, nominal, central)

        return central, summary
