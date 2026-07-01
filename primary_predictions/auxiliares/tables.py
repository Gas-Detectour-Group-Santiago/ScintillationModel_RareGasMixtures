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

def export_normalization_comparison_table(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    left_prefix: str = "arcf4_norm",
    right_prefix: str = "arn2_norm",
    left_heading: str = r"Norm. Ar--CF$_4$",
    right_heading: str = r"Norm. Ar--N$_2$",
) -> None:
    """Export selected predictions under two reference normalizations.

    The input dataframe is expected to contain columns named like
    ``value_<prefix>``, ``stat_minus_<prefix>``, ``stat_plus_<prefix>``,
    ``syst_minus_<prefix>`` and ``syst_plus_<prefix>`` for each prefix.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def col(kind: str, prefix: str) -> str:
        return f"{kind}_{prefix}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        rf" & \multicolumn{{3}}{{c}}{{{left_heading}}} & \multicolumn{{3}}{{c}}{{{right_heading}}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Predicción & Valor & Stat. & Syst. & Valor & Stat. & Syst. \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"{row['tex_label']} & "
            f"{fmt_num(row[col('value', left_prefix)])} & "
            f"{fmt_asym(row[col('stat_minus', left_prefix)], row[col('stat_plus', left_prefix)])} & "
            f"{fmt_asym(row[col('syst_minus', left_prefix)], row[col('syst_plus', left_prefix)])} & "
            f"{fmt_num(row[col('value', right_prefix)])} & "
            f"{fmt_asym(row[col('stat_minus', right_prefix)], row[col('stat_plus', right_prefix)])} & "
            f"{fmt_asym(row[col('syst_minus', right_prefix)], row[col('syst_plus', right_prefix)])} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")

