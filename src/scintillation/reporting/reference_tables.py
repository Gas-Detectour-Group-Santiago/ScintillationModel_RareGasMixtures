from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


def _num(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(x):
        return "--"
    return rf"\num{{{x:.5g}}}"


def export_second_continuum_parameter_table(project_root: str | Path) -> Path:
    root = Path(project_root)
    source = root / "data" / "reference" / "parameters" / "Ar2nd_continium.csv"
    frame = pd.read_csv(source)
    active = frame.loc[
        frame["enabled"].astype(str).str.lower().isin({"1", "true", "yes", "on"})
        & pd.to_numeric(frame["value"], errors="coerce").notna()
    ].copy()
    out = root / "outputs" / "tables" / "reference" / "argon_second_continuum_parameters.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{longtable}{lllll}",
        r"\caption{Parámetros activos del modelo general del segundo continuo de argón. Los parámetros comunes se aplican a todas las mezclas; los parámetros de aditivo se seleccionan por gas.}\label{tab:argon_second_continuum_parameters}\\",
        r"\toprule",
        r"Ámbito & Aditivo & Parámetro & Valor & Referencia \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Ámbito & Aditivo & Parámetro & Valor & Referencia \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in active.itertuples(index=False):
        scope = "común" if str(row.scope) == "common" else "aditivo"
        additive = "--" if pd.isna(row.additive) or str(row.additive).strip() == "" else str(row.additive)
        tex_name = str(row.tex_name) if not pd.isna(row.tex_name) else str(row.name)
        lines.append(f"{scope} & {additive} & {tex_name} & {_num(row.value)} {row.unit} & {row.reference} \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
