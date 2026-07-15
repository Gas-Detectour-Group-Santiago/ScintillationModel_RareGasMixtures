from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def export_n2_pure_gaussian_mean_spectrum_table(
    *,
    primary_prediction_csv: Path,
    summary_ratio_csv: Path,
    output_csv: Path,
    output_tex: Path,
    caption: str,
    label: str,
    gaussian_ratio_name: str = "ArN2_mean_gaussian_over_ArCF4_95_5_VIS",
    hardcut_ratio_name: str = "ArN2_mean_hardcut_over_ArCF4_95_5_VIS",
) -> pd.DataFrame:
    """Apply the mean-spectrum Gaussian correction to pure-N2 predictions.

    The primary-prediction table is the hardcut/standard reference.  The
    Gaussian table keeps the same three Degrad entries but multiplies both
    normalisations and their propagated stat/sys uncertainties by the integral
    factor:

        R(mean spectrum, gaussian) / R(mean spectrum, hardcut).
    """
    primary_prediction_csv = Path(primary_prediction_csv)
    summary_ratio_csv = Path(summary_ratio_csv)
    if not primary_prediction_csv.exists():
        raise FileNotFoundError(f"Missing primary prediction table: {primary_prediction_csv}")
    if not summary_ratio_csv.exists():
        raise FileNotFoundError(f"Missing integral summary table: {summary_ratio_csv}")

    primary = pd.read_csv(primary_prediction_csv)
    summary = pd.read_csv(summary_ratio_csv)
    scale = _gaussian_over_hardcut_scale(summary, gaussian_ratio_name, hardcut_ratio_name)

    out = primary.copy()
    out["gaussian_ratio_name"] = gaussian_ratio_name
    out["hardcut_ratio_name"] = hardcut_ratio_name
    out["gaussian_over_hardcut_scale"] = scale

    for prefix in ("arcf4_norm", "arn2_norm"):
        for kind in ("value", "stat_minus", "stat_plus", "syst_minus", "syst_plus", "total_minus", "total_plus"):
            col = f"{kind}_{prefix}"
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce") * scale

    output_csv = Path(output_csv)
    output_tex = Path(output_tex)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    _write_latex(out, output_tex, caption=caption, label=label, scale=scale)
    return out


def _gaussian_over_hardcut_scale(summary: pd.DataFrame, gaussian_ratio_name: str, hardcut_ratio_name: str) -> float:
    names = summary["case"].astype(str) if "case" in summary.columns else summary["ratio_name"].astype(str)
    gauss = summary.loc[names.eq(gaussian_ratio_name)]
    hard = summary.loc[names.eq(hardcut_ratio_name)]
    if gauss.empty or hard.empty:
        available = ", ".join(names.tolist())
        raise KeyError(
            f"Cannot find gaussian/hardcut ratio names. Requested {gaussian_ratio_name!r} and "
            f"{hardcut_ratio_name!r}. Available: {available}"
        )
    g = float(gauss.iloc[0]["ratio"])
    h = float(hard.iloc[0]["ratio"])
    if not np.isfinite(g) or not np.isfinite(h) or h == 0.0:
        raise ValueError(f"Invalid gaussian/hardcut ratios: gaussian={g}, hardcut={h}")
    return g / h


def _write_latex(df: pd.DataFrame, tex_path: Path, *, caption: str, label: str, scale: float) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Norm. Ar--CF$_4$} & \multicolumn{3}{c}{Norm. Ar--N$_2$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Entrada Degrad & Valor & Stat. & Syst. & Valor & Stat. & Syst. \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['tex_label']} & "
            f"{_fmt_num(row['value_arcf4_norm'])} & "
            f"{_fmt_asym(row['stat_minus_arcf4_norm'], row['stat_plus_arcf4_norm'])} & "
            f"{_fmt_asym(row['syst_minus_arcf4_norm'], row['syst_plus_arcf4_norm'])} & "
            f"{_fmt_num(row['value_arn2_norm'])} & "
            f"{_fmt_asym(row['stat_minus_arn2_norm'], row['stat_plus_arn2_norm'])} & "
            f"{_fmt_asym(row['syst_minus_arn2_norm'], row['syst_plus_arn2_norm'])} " r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\par\vspace{{0.35em}}\footnotesize{{Corrección gaussiana mean-spectrum aplicada: $R_G/R_{{HC}}={_fmt_num(scale)}$.}}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_num(x: object, sig: int = 3) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return r"--"
    if not np.isfinite(value):
        return r"--"
    return rf"\num{{{value:.{sig}g}}}"


def _fmt_asym(minus: object, plus: object, sig: int = 2) -> str:
    try:
        m = float(minus)
        p = float(plus)
    except (TypeError, ValueError):
        return r"--"
    if not np.isfinite(m) or not np.isfinite(p):
        return r"--"
    if abs(m) == 0 and abs(p) == 0:
        return r"--"
    return rf"$^{{+{_fmt_num(p, sig)}}}_{{-{_fmt_num(m, sig)}}}$"
