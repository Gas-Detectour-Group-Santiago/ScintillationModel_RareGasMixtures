from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import pandas as pd

from spectra import config as cfg
from spectra.auxiliares import (
    build_generated_spectra,
    find_project_root,
    output_dir,
    run_annotated_figures,
    export_vuv_prediction_tables,
)
from spectra.auxiliares.generated import build_generated_amplied_spectra
from spectra.auxiliares.raw import raw_aggregated_dataframe, raw_mosaic_dataframe, raw_reference_dataframe
from spectra.auxiliares.comparison import comparison_dataframe
from spectra.auxiliares.common import ensure_parent

from scintillation.plotting.recipe_config import active_spectrum_recipes, as_bool, as_text, split_values
from scintillation.plotting.spectra_recipe_engine import render_raw, render_generated, render_comparison


def _floats(value: object, fallback: tuple[float, ...]) -> tuple[float, ...]:
    parsed = split_values(value, cast=float)
    return parsed or fallback


def _union(rows: pd.DataFrame, column: str, fallback: tuple[float, ...]) -> tuple[float, ...]:
    values: set[float] = set()
    if column in rows:
        for value in rows[column]:
            values.update(_floats(value, ()))
    return tuple(sorted(values)) or fallback


def _prepare_generation_grid(recipes: pd.DataFrame) -> None:
    generated = recipes.loc[recipes["plot_type"].isin(["generated", "generated_extended", "comparison"])]
    cfg.GENERATED_PRESSURES_BAR = _union(generated, "pressures_bar", tuple(cfg.GENERATED_PRESSURES_BAR))
    cfg.GENERATED_CONCENTRATIONS_PERCENT = _union(
        generated, "concentrations_percent", tuple(cfg.GENERATED_CONCENTRATIONS_PERCENT)
    )


def _raw_data(project_root: Path, outdir: Path, recipes: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    all_raw: dict[str, pd.DataFrame] = {}
    gases = tuple(dict.fromkeys(as_text(value) for value in recipes.get("gas", ()) if as_text(value)))
    for gas in gases:
        all_columns = raw_aggregated_dataframe(project_root, gas)
        if all_columns.empty:
            print(f"[spectra-recipes] warning: no raw spectra for {gas}")
            continue
        path = outdir / "csv" / f"{gas}_raw_spectra_aggregated_C1_C2_mean.csv.gz"
        ensure_parent(path)
        all_columns.to_csv(path, index=False, compression="gzip")
        (outdir / "csv" / f"{gas}_raw_spectra_aggregated_C1_C2_mean.csv").unlink(missing_ok=True)
        all_raw[gas] = all_columns
    return all_raw, raw_reference_dataframe(all_raw)


def _render_raw(project_root: Path, outdir: Path, recipes: pd.DataFrame) -> None:
    if recipes.empty:
        return
    all_raw, reference = _raw_data(project_root, outdir, recipes)
    for _, row in recipes.iterrows():
        gas = as_text(row.get("gas"))
        if gas not in all_raw:
            continue
        cfg.RAW_PLOT_SPECTRUM_COLUMN = as_text(row.get("components"), "mean_spectrum")
        cfg.RAW_PRESSURES_BAR = _floats(row.get("pressures_bar"), tuple(cfg.RAW_PRESSURES_BAR))
        cfg.RAW_CONCENTRATIONS_PERCENT = _floats(
            row.get("concentrations_percent"), tuple(cfg.RAW_CONCENTRATIONS_PERCENT)
        )
        frame = raw_mosaic_dataframe(all_raw[gas], gas)
        render_raw(row, project_root=project_root, frame=frame, reference=reference)


def _render_standard_generated(project_root: Path, outdir: Path, recipes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    standard_rows = recipes.loc[recipes["plot_type"] == "generated"]
    if standard_rows.empty:
        return {}
    compact: dict[str, pd.DataFrame] = {}
    gases = tuple(dict.fromkeys(as_text(value) for value in standard_rows["gas"] if as_text(value)))
    for gas in gases:
        gas_rows = standard_rows.loc[standard_rows["gas"].astype(str) == gas]
        components = tuple(
            dict.fromkeys(
                component
                for value in gas_rows["components"]
                for component in (split_values(value, cast=str) or ("total",))
            )
        )
        generated = build_generated_spectra(project_root, outdir, gases=[gas], components=components)
        frame = generated[gas]
        for _, row in standard_rows.loc[standard_rows["gas"].astype(str) == gas].iterrows():
            render_generated(row, project_root=project_root, frame=frame)
        compact[gas] = frame.loc[frame["component"].astype(str) == "total"].copy() if "component" in frame else frame
        del generated, frame
        gc.collect()
    return compact


def _render_extended_generated(project_root: Path, outdir: Path, recipes: pd.DataFrame) -> None:
    extended_rows = recipes.loc[recipes["plot_type"] == "generated_extended"]
    if extended_rows.empty:
        return
    gases = tuple(dict.fromkeys(as_text(value) for value in extended_rows["gas"] if as_text(value)))
    for gas in gases:
        gas_rows = extended_rows.loc[extended_rows["gas"].astype(str) == gas]
        components = tuple(
            dict.fromkeys(
                component
                for value in gas_rows["components"]
                for component in (split_values(value, cast=str) or ("total",))
            )
        )
        generated = build_generated_amplied_spectra(
            project_root, outdir, gases=[gas], components=components
        )
        frame = generated[gas]
        for _, row in extended_rows.loc[extended_rows["gas"].astype(str) == gas].iterrows():
            render_generated(row, project_root=project_root, frame=frame)
        del generated, frame
        gc.collect()
    export_vuv_prediction_tables(project_root)


def _render_comparisons(project_root: Path, outdir: Path, recipes: pd.DataFrame,
                        generated: dict[str, pd.DataFrame]) -> None:
    if recipes.empty:
        return
    if not generated:
        needed_gases = tuple(
            dict.fromkeys(
                gas
                for value in recipes.get("gas", ())
                for gas in (split_values(value, cast=str) or (as_text(value),))
                if gas
            )
        )
        full = build_generated_spectra(
            project_root, outdir, gases=list(needed_gases), components=("total",)
        )
        generated = {
            gas: frame.loc[frame["component"].astype(str) == "total"].copy()
            if "component" in frame else frame
            for gas, frame in full.items()
        }
        del full
        gc.collect()
    for _, row in recipes.iterrows():
        gases = split_values(row.get("gas"), cast=str) or (as_text(row.get("gas")),)
        pressures = _floats(row.get("pressures_bar"), tuple(cfg.COMPARISON_PRESSURES_BAR))
        concentrations = _floats(row.get("concentrations_percent"), tuple(cfg.COMPARISON_CONCENTRATIONS_PERCENT))
        spec = {
            "name": as_text(row.get("plot_id")),
            "title": as_text(row.get("title")),
            "gases": gases,
            "pressures_bar": pressures,
            "concentrations_percent": concentrations,
            "output_pdf": Path(as_text(row.get("output"))).name,
            "output_csv": f"{as_text(row.get('plot_id'))}.csv",
        }
        cfg.WAVELENGTH_RANGE_COMPARISON_NM = (
            float(row.get("wavelength_min_nm") or 180.0),
            float(row.get("wavelength_max_nm") or 800.0),
        )
        frame = comparison_dataframe(project_root, generated, spec)
        cache_path = outdir / "csv" / f"{Path(spec["output_csv"]).stem}.csv.gz"
        ensure_parent(cache_path)
        frame.to_csv(cache_path, index=False, compression="gzip")
        render_comparison(row, project_root=project_root, frame=frame)
        del frame
        gc.collect()


def _run_internal_group(project_root: Path, outdir: Path, recipes: pd.DataFrame, group: str) -> None:
    if group == "raw":
        _render_raw(project_root, outdir, recipes.loc[recipes["plot_type"] == "raw"])
        annotated = recipes.loc[recipes["plot_type"] == "annotated"]
        if not annotated.empty:
            run_annotated_figures(project_root, outdir, annotated)
        return

    kind, gas = group.split(":", 1)
    selected = recipes.loc[recipes["gas"].astype(str) == gas]
    if kind == "standard":
        generated_rows = selected.loc[selected["plot_type"] == "generated"]
        comparison_rows = selected.loc[selected["plot_type"] == "comparison"]
        generated = _render_standard_generated(project_root, outdir, generated_rows)
        _render_comparisons(project_root, outdir, comparison_rows, generated)
        return
    if kind == "extended":
        _render_extended_generated(
            project_root,
            outdir,
            selected.loc[selected["plot_type"] == "generated_extended"],
        )
        return
    raise ValueError(f"Unknown internal spectra group: {group}")


def main(*, internal_group: str | None = None) -> None:
    project_root = find_project_root(Path(__file__))
    outdir = output_dir(project_root)
    recipes = active_spectrum_recipes(project_root)
    if recipes.empty:
        raise RuntimeError("No active spectra recipes found in config/plots/spectra.csv")
    _prepare_generation_grid(recipes)

    if internal_group is not None:
        _run_internal_group(project_root, outdir, recipes, internal_group)
        return

    # Isolate large gas grids in short-lived processes. This keeps peak memory
    # bounded and prevents allocator fragmentation from making the second gas
    # dramatically slower than the first one.
    groups = ["raw"]
    standard_gases = tuple(
        dict.fromkeys(
            as_text(value)
            for value in recipes.loc[recipes["plot_type"].isin(["generated", "comparison"]), "gas"]
            if as_text(value)
        )
    )
    # Comparison rows normally contain one gas. Split multi-gas text if it is
    # introduced later without changing the public CSV schema.
    expanded_standard: list[str] = []
    for value in standard_gases:
        expanded_standard.extend(split_values(value, cast=str) or (value,))
    groups.extend(f"standard:{gas}" for gas in dict.fromkeys(expanded_standard))
    extended_gases = tuple(
        dict.fromkeys(
            as_text(value)
            for value in recipes.loc[recipes["plot_type"] == "generated_extended", "gas"]
            if as_text(value)
        )
    )
    groups.extend(f"extended:{gas}" for gas in extended_gases)

    for group in groups:
        print(f"[spectra] isolated group {group}")
        subprocess.run(
            [sys.executable, Path(__file__).name, "--internal-group", group],
            cwd=Path(__file__).parent,
            check=True,
        )


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CSV-driven spectra generator")
    parser.add_argument("--internal-group", default=None, help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args(argv)
    main(internal_group=args.internal_group)


if __name__ == "__main__":
    cli()
