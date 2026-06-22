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

