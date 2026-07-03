from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fmt_num(x: float, sig: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return r"--"
    return rf"\num{{{float(x):.{sig}g}}}"


def fmt_asym(minus: float, plus: float, sig: int = 2) -> str:
    if not np.isfinite(minus) or not np.isfinite(plus):
        return r"--"
    if abs(minus) == 0 and abs(plus) == 0:
        return r"--"
    return rf"$^{{+{fmt_num(plus, sig)}}}_{{-{fmt_num(minus, sig)}}}$"


# Preferred table names used in the TFM text.  Internal parameter names remain
# unchanged so model evaluation, OCW rules and existing CSVs stay compatible.
PARAMETER_TEX_NAME_ALIASES = {
    "Nnorm": r"$N_{\mathrm{norm}}$",
    # Ar--CF4 UV/VIS optical-conversion weights and branching ratio.
    "P_CF3_vis_dir": r"$\mathcal{W}_{\mathrm{CF}_3^*,\mathrm{vis}}^{\mathrm{dir}}$",
    "P_CF4_dir": r"$\mathcal{W}_{\mathrm{CF}_4^{+,*}}^{\mathrm{dir}}$",
    "P_Ar3rd": r"$\mathcal{W}_{\mathrm{Ar}^{++}}$",
    "P_CF3_uv_dir": r"$\mathcal{B}r_{\mathrm{UV}}(\mathrm{CF}_3^*)$",
    "K_Ar_dbleStar_Q_Ar": r"$K_{\mathrm{Ar}^{**},Q(\mathrm{Ar})}\,[\mathrm{ns}^{-1}]$",
    "K_Ar_dbleStar_Q_CF4": r"$K_{\mathrm{Ar}^{**},Q(\mathrm{CF}_4)}\,[\mathrm{ns}^{-1}]$",
    "K_Ar3rd_Q_CF4": r"$K_{\mathrm{Ar}^{++},Q(\mathrm{CF}_4)}\,[\mathrm{ns}^{-1}]$",
    # Ar--N2 UV model.
    "P_N2": r"$\mathcal{W}_{\mathrm{N}_2^*}$",
    "tau_N2": r"$\tau_{\mathrm{N}_2(C)}\,[\mathrm{ns}]$",
    "K_N2_Q_N2": r"$K_{\mathrm{N}_2(C)Q(\mathrm{N}_2)}\,[\mathrm{ns}^{-1}]$",
    "K_N2_Q_Ar": r"$K_{\mathrm{N}_2(C)Q(\mathrm{Ar})}\,[\mathrm{ns}^{-1}]$",
    "K_ArMeta_Q_N2c": r"$K_{\mathrm{Ar}(1s_5)Q(\mathrm{N}_2(C))}\,[\mathrm{ns}^{-1}]$",
    "K_ArMeta_Q_N2b": r"$K_{\mathrm{Ar}(1s_5)Q(\mathrm{N}_2(B))}\,[\mathrm{ns}^{-1}]$",
    "K_ArMeta_Q_2Ar": r"$K_{\mathrm{Ar}(1s_5)Q(2\mathrm{Ar})}\,[\mathrm{ns}^{-1}]$",
    "K_ArRes_Q_N2c": r"$K_{\mathrm{Ar}(1s_4)Q(\mathrm{N}_2(C))}\,[\mathrm{ns}^{-1}]$",
    "K_ArRes_Q_N2b": r"$K_{\mathrm{Ar}(1s_4)Q(\mathrm{N}_2(B))}\,[\mathrm{ns}^{-1}]$",
    "K_ArRes_Q_2Ar": r"$K_{\mathrm{Ar}(1s_4)Q(2\mathrm{Ar})}\,[\mathrm{ns}^{-1}]$",
    "frac_Ar_dbleStar": r"$f_{\mathrm{Ar}^{**}}$",
}

FIT_PARAMETER_TEX_NAME_ALIASES = {
    ("ArCF4_primary", "P_Ar_dbleStar"): r"$\mathcal{W}_{\mathrm{Ar}^{**}}$",
    ("ArN2_primary", "P_Ar_dbleStar"): r"$\mathcal{W}_{\mathrm{Ar}^{**}}^*$",
}


def _line_suffix(name: str, prefix: str) -> str | None:
    if not name.startswith(prefix):
        return None
    return name[len(prefix):]


def preferred_parameter_tex_name(name: str, fallback: str | None = None, *, fit_name: str | None = None) -> str:
    """Return the display label used by the current TFM notation."""
    if fit_name is not None and (fit_name, name) in FIT_PARAMETER_TEX_NAME_ALIASES:
        return FIT_PARAMETER_TEX_NAME_ALIASES[(fit_name, name)]
    if name in PARAMETER_TEX_NAME_ALIASES:
        return PARAMETER_TEX_NAME_ALIASES[name]

    for prefix in ("PAr_star_",):
        line = _line_suffix(name, prefix)
        if line:
            return rf"$\mathcal{{W}}_{{\mathrm{{Ar}}^{{**}},{line}\,\mathrm{{nm}}}}$"

    for prefix in ("tau_CF4_", "tau_N2_"):
        line = _line_suffix(name, prefix)
        if line:
            return rf"$\tau_{{\mathrm{{Ar}}^{{**}},{line}\,\mathrm{{nm}}}}$"

    line = _line_suffix(name, "K_Ar_Q_Ar_")
    if line:
        return rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{Ar}}),{line}\,\mathrm{{nm}}}}$"

    line = _line_suffix(name, "K_Ar_Q_CF4_")
    if line:
        return rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{CF}}_4),{line}\,\mathrm{{nm}}}}$"

    line = _line_suffix(name, "K_Ar_Q_N2_")
    if line:
        return rf"$K_{{\mathrm{{Ar}}^{{**}}Q(\mathrm{{N}}_2),{line}\,\mathrm{{nm}}}}$"

    return str(fallback or name)


def export_prediction_table(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    value_col: str = "value",
    stat_cols: tuple[str, str] = ("stat_minus", "stat_plus"),
    syst_cols: tuple[str, str] = ("syst_minus", "syst_plus"),
    total_cols: tuple[str, str] = ("total_minus", "total_plus"),
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Predicción & Valor & Stat. & Syst. & Total \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"{row['tex_label']} & {fmt_num(row[value_col])} & "
            f"{fmt_asym(row[stat_cols[0]], row[stat_cols[1]])} & "
            f"{fmt_asym(row[syst_cols[0]], row[syst_cols[1]])} & "
            f"{fmt_asym(row[total_cols[0]], row[total_cols[1]])} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")



# -----------------------------------------------------------------------------
# Secondary parameter tables
# -----------------------------------------------------------------------------

def _fmt_latex_cell(x: float | None, sig: int = 3) -> str:
    return fmt_num(float(x), sig=sig) if x is not None and np.isfinite(float(x)) else r"--"


def _parameter_delta_is_nontrivial(value: float, optimum: float) -> bool:
    if not (np.isfinite(value) and np.isfinite(optimum)):
        return False
    return not np.isclose(float(value), float(optimum), rtol=1e-12, atol=1e-15)


def _load_primary_parameter_summary(project_root: Path, fit_name: str) -> pd.DataFrame:
    """Read the primary parameter summary exported by primary_fits.

    Preferred input is data/Parameters/<fit_name>.csv, because it already
    contains the parameter names, TeX labels and toy-derived stat./syst. bands.
    A minimal fallback to data/FitResults/<fit_name>_central.csv is kept so the
    secondary script does not fail in older worktrees.
    """
    project_root = Path(project_root)
    candidates = [
        project_root / "data" / "Parameters" / f"{fit_name}.csv",
        project_root / "data" / "FitResults" / f"{fit_name}_central.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"name", "value"}.issubset(df.columns):
            out = df.copy()
            if "tex_name" not in out.columns:
                out["tex_name"] = out["name"]
            for col in ("stat_minus", "stat_plus", "syst_minus", "syst_plus"):
                if col not in out.columns:
                    out[col] = np.nan
            return out
        if {"parameter_name", "value"}.issubset(df.columns):
            out = df.rename(columns={"parameter_name": "name"}).copy()
            if "tex_name" not in out.columns:
                out["tex_name"] = out["name"]
            for col in ("stat_minus", "stat_plus", "syst_minus", "syst_plus"):
                if col not in out.columns:
                    out[col] = np.nan
            return out

    raise FileNotFoundError(
        f"No encuentro la tabla primaria de parámetros para {fit_name!r}. "
        "Ejecuta antes primary_fits o revisa data/Parameters."
    )


def _iter_band_curves(obj):
    """Yield every simple BandCurveConfig found inside a plot/combined config."""
    curves = getattr(obj, "curves", None)
    if curves is None:
        if hasattr(obj, "fit_name"):
            yield obj
        return
    for curve in curves:
        yield from _iter_band_curves(curve)


def _collect_secondary_rules(plot_configs) -> dict[str, dict[str, object]]:
    """Collect OCW/secondary optimum rules by fit name.

    The returned structure is fit_name -> parameter_name -> rule.  Fit names
    used without secondary rules are still included, so they also get a table.
    """
    by_fit: dict[str, dict[str, object]] = {}
    for plot in plot_configs:
        for curve in _iter_band_curves(plot):
            fit_name = getattr(curve, "fit_name", None)
            if not fit_name:
                continue
            by_fit.setdefault(str(fit_name), {})
            ocw_config = getattr(curve, "ocw_config", None)
            if ocw_config is None:
                continue
            for rule in getattr(ocw_config, "rules", ()):
                by_fit[str(fit_name)][str(rule.name)] = rule
    return by_fit


def _build_secondary_parameter_table(project_root: Path, fit_name: str, rules_by_name: dict[str, object]) -> pd.DataFrame:
    primary = _load_primary_parameter_summary(project_root, fit_name)
    rows: list[dict[str, object]] = []

    for _, row in primary.iterrows():
        name = str(row["name"])
        value = float(row["value"])
        rule = rules_by_name.get(name)
        # The secondary column is the parameter value actually used by the
        # secondary optimum.  If there is no secondary/OCW rule, that value is
        # exactly the primary fitted value, not a dash.
        optimal = value
        modified = False
        if rule is not None:
            optimal = float(rule.apply(value, "optimum"))
            modified = _parameter_delta_is_nontrivial(value, optimal)

        rows.append(
            {
                "parameter": name,
                "tex_name": preferred_parameter_tex_name(name, row.get("tex_name", name), fit_name=fit_name),
                "primary_value": value,
                "primary_stat_minus": float(row.get("stat_minus", np.nan)),
                "primary_stat_plus": float(row.get("stat_plus", np.nan)),
                "primary_syst_minus": float(row.get("syst_minus", np.nan)),
                "primary_syst_plus": float(row.get("syst_plus", np.nan)),
                "secondary_optimal": optimal,
                "secondary_has_rule": rule is not None,
                "secondary_modified": modified,
            }
        )

    return pd.DataFrame(rows)


def export_secondary_parameter_table(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
) -> None:
    """Export parameter table with primary value/stat/syst and secondary optimum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Primary} & \multicolumn{1}{c}{Secondary} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}",
        r"Parameter & Value & Stat. & Syst. & Optimal \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        optimal = row["secondary_optimal"]
        lines.append(
            f"{row['tex_name']} & {_fmt_latex_cell(row['primary_value'])} & "
            f"{fmt_asym(row['primary_stat_minus'], row['primary_stat_plus'])} & "
            f"{fmt_asym(row['primary_syst_minus'], row['primary_syst_plus'])} & "
            f"{_fmt_latex_cell(optimal)} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


FIT_DISPLAY_NAMES = {
    "ArCF4_primary": r"Ar--CF$_4$ primario",
    "ArCF4_IR_primary": r"Ar--CF$_4$ IR primario",
    "ArN2_primary": r"Ar--N$_2$ primario",
    "ArN2_IR_primary": r"Ar--N$_2$ IR primario",
}


def fit_display_name(fit_name: str) -> str:
    return FIT_DISPLAY_NAMES.get(str(fit_name), str(fit_name).replace("_", r"\_"))


def export_secondary_parameter_tables(
    project_root: Path,
    plot_configs,
    *,
    tables_subdir: str = "param_secondary",
    extra_fit_names: tuple[str, ...] | list[str] = (),
) -> dict[str, dict[str, Path]]:
    """Export one duplicated primary/secondary parameter table per fit.

    Tables are written under data/Tables/param_secondary/.  If a parameter has
    an OCW/secondary rule, Secondary/Optimal is the rule optimum.  Otherwise it
    repeats the primary value, which makes clear that secondary uses the same
    parameter.
    """
    project_root = Path(project_root)
    rules_by_fit = _collect_secondary_rules(plot_configs)
    for fit_name in extra_fit_names:
        rules_by_fit.setdefault(str(fit_name), {})
    if not rules_by_fit:
        return {}

    outdir = project_root / "data" / "Tables" / tables_subdir
    outdir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, Path]] = {}
    missing: dict[str, str] = {}
    for fit_name in sorted(rules_by_fit):
        try:
            df = _build_secondary_parameter_table(project_root, fit_name, rules_by_fit[fit_name])
        except FileNotFoundError as exc:
            missing[fit_name] = str(exc)
            continue
        csv_path = outdir / f"{fit_name}_param_secondary.csv"
        tex_path = outdir / f"{fit_name}_param_secondary.tex"
        df.to_csv(csv_path, index=False)
        export_secondary_parameter_table(
            df,
            tex_path,
            caption=f"Valores de los parámetros primarios y secundarios para {fit_display_name(fit_name)}.",
            label=f"tab:{fit_name}_param_secondary",
        )
        outputs[fit_name] = {"csv": csv_path, "tex": tex_path}

    return outputs
