from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_num(x: float) -> str:
    if x is None or not np.isfinite(x):
        return r"--"
    return rf"\num{{{float(x):.3g}}}"


def _fmt_asym(m: float, p: float) -> str:
    if (not np.isfinite(m)) or (not np.isfinite(p)):
        return r"--"
    if abs(m) == 0 and abs(p) == 0:
        return r"--"
    return rf"$^{{+{_fmt_num(p)}}}_{{-{_fmt_num(m)}}}$"


def build_parameter_summary(
    names: list[str],
    tex_names: list[str],
    central: np.ndarray,
    central_err: np.ndarray,
    stat_minus: np.ndarray,
    stat_plus: np.ndarray,
    syst_minus: np.ndarray,
    syst_plus: np.ndarray,
    fixed_idx: list[int],
) -> pd.DataFrame:
    central = np.asarray(central, dtype=float)
    total_minus = np.sqrt(np.asarray(stat_minus) ** 2 + np.asarray(syst_minus) ** 2)
    total_plus = np.sqrt(np.asarray(stat_plus) ** 2 + np.asarray(syst_plus) ** 2)

    return pd.DataFrame(
        {
            "name": names,
            "tex_name": tex_names,
            "value": central,
            "fit_uncertainty": central_err,
            "stat_minus": stat_minus,
            "stat_plus": stat_plus,
            "syst_minus": syst_minus,
            "syst_plus": syst_plus,
            "total_minus": total_minus,
            "total_plus": total_plus,
            "fixed": [i in set(fixed_idx) for i in range(len(names))],
        }
    )


def export_stat_syst_latex(
    summary: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Parámetro & Valor & Stat. & Syst. & Total \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{row['tex_name']} & {_fmt_num(row['value'])} & "
            f"{_fmt_asym(row['stat_minus'], row['stat_plus'])} & "
            f"{_fmt_asym(row['syst_minus'], row['syst_plus'])} & "
            f"{_fmt_asym(row['total_minus'], row['total_plus'])} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_toys(path: Path, names: list[str], toys: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if toys.size == 0:
        pd.DataFrame(columns=["toy_id"] + names).to_csv(path, index=False)
        return
    df = pd.DataFrame(toys, columns=names)
    df.insert(0, "toy_id", np.arange(len(df), dtype=int))
    df.to_csv(path, index=False)


def export_vector(path: Path, names: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": names, "value": values}).to_csv(path, index=False)


def export_matrix(path: Path, names: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path)
