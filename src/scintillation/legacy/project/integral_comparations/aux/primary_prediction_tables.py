from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrimaryPredictionReference:
    """Reference prediction used to convert integral-summary rows into absolute yields.

    The default input is the wide CSV written by primary predictions:

        data/Predictions/primary_selected_yields_arcf4_vs_arn2_norm.csv

    It contains two parallel normalisations for each selected prediction:
    ``*_arcf4_norm`` and ``*_arn2_norm``.  The reference row is selected by
    ``reference_id`` so the integral module can later use a VIS, IR or any other
    denominator without changing the implementation.
    """

    csv_path: Path
    reference_id: str
    fallback_ids: tuple[str, ...] = ()
    forced_tex_label: str | None = None
    arcf4_label: str = r"Norm. Ar--CF$_4$"
    arn2_label: str = r"Norm. Ar--N$_2$"
    scale_mode: Literal["direct_ratio", "relative_to_anchor"] = "direct_ratio"
    anchor_ratio_name: str | None = None
    anchor_case_label: str = "mean, hardcut"


@dataclass(frozen=True)
class AbsolutePredictionTableConfig:
    caption: str
    label: str
    unit: str = r"ph/MeV"
    include_ratio_column: bool = False
    use_total_uncertainty: bool = True
    digits: int = 3


@dataclass(frozen=True)
class NormalizedReferenceValue:
    value: float
    stat_minus: float
    stat_plus: float
    syst_minus: float
    syst_plus: float
    total_minus: float
    total_plus: float


def build_absolute_prediction_table(
    ratio_summary: pd.DataFrame,
    reference: PrimaryPredictionReference,
    *,
    ratio_titles: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Convert integral-summary rows into absolute predictions using one primary anchor.

    Parameters
    ----------
    ratio_summary:
        Summary rows produced by ``build_summary_table``.  They must contain one
        row per ratio case and at least ``ratio`` and ``ratio_name``.
    reference:
        Primary prediction CSV plus selected reference id.
    ratio_titles:
        Optional map used to turn technical ratio names into table labels.

    Returns
    -------
    table, metadata
        ``table`` has one row per ratio case and both normalisations already
        propagated. With ``scale_mode="direct_ratio"``, the multiplier is the
        summary ratio itself. With ``scale_mode="relative_to_anchor"``, the
        multiplier is ``ratio / anchor_ratio``. The propagated uncertainties are
        those of the selected primary anchor scaled by this multiplier; the
        current integral ratios do not carry their own statistical uncertainty.
    """

    if ratio_summary.empty:
        return ratio_summary.copy(), {"status": "empty_summary"}

    ref_df = pd.read_csv(reference.csv_path)
    requested_reference_id = str(reference.reference_id)
    candidate_ids = (requested_reference_id, *tuple(str(x) for x in reference.fallback_ids))

    matched_reference_id: str | None = None
    ref_rows = ref_df.iloc[0:0].copy()
    for candidate_id in candidate_ids:
        ref_rows = ref_df[ref_df["id"].astype(str).eq(candidate_id)].copy()
        if not ref_rows.empty:
            matched_reference_id = candidate_id
            break

    if ref_rows.empty or matched_reference_id is None:
        available = ", ".join(ref_df["id"].astype(str).tolist())
        raise KeyError(
            f"Reference prediction {reference.reference_id!r} not found in {reference.csv_path}. "
            f"Fallback ids tried: {list(reference.fallback_ids)}. Available ids: {available}"
        )
    ref_row = ref_rows.iloc[0]
    reference_tex_label = reference.forced_tex_label or ref_row.get("tex_label", ref_row.get("label", matched_reference_id))
    reference_label = reference.forced_tex_label or ref_row.get("label", matched_reference_id)

    arcf4_ref = _reference_from_wide_row(ref_row, "arcf4_norm")
    arn2_ref = _reference_from_wide_row(ref_row, "arn2_norm")

    ratio_titles = ratio_titles or {}
    anchor_ratio = _resolve_anchor_ratio(ratio_summary, reference, ratio_titles)

    rows: list[dict[str, object]] = []

    for _, row in ratio_summary.iterrows():
        ratio = float(row.get("ratio", np.nan))
        if not np.isfinite(ratio):
            continue

        ratio_name = str(row.get("case", row.get("ratio_name", "")))
        if ratio_name not in ratio_titles and "ratio_name" in row:
            ratio_name = str(row["ratio_name"])
        case_label = ratio_titles.get(ratio_name, _default_case_label(row))
        scale_factor = _scale_factor_from_ratio(ratio, anchor_ratio, reference.scale_mode)
        if not np.isfinite(scale_factor):
            continue

        out = {
            "case": case_label,
            "ratio_name": ratio_name,
            "spectrum_column": row.get("spectrum_column", row.get("numerator_spectrum_column", "")),
            "method": row.get("method", row.get("numerator_method", "")),
            "pressure_bar": row.get("pressure_bar", np.nan),
            "concentration_percent": row.get("concentration_percent", np.nan),
            "ratio": ratio,
            "anchor_ratio": anchor_ratio,
            "scale_mode": reference.scale_mode,
            "scale_factor": scale_factor,
            "reference_id": requested_reference_id,
            "matched_reference_id": matched_reference_id,
            "reference_label": reference_label,
            "reference_tex_label": reference_tex_label,
            "unit": ref_row.get("unit", ""),
        }
        out.update(_scaled_columns("arcf4_norm", scale_factor, arcf4_ref))
        out.update(_scaled_columns("arn2_norm", scale_factor, arn2_ref))
        rows.append(out)

    table = pd.DataFrame(rows)
    if table.empty:
        return table, {"status": "empty_after_filter", "reference_id": reference.reference_id}

    if ratio_titles:
        order = {str(name): idx for idx, name in enumerate(ratio_titles.keys())}
        table["_ratio_order"] = table["ratio_name"].astype(str).map(order).fillna(len(order)).astype(int)
        table = table.sort_values(["_ratio_order", "case"]).drop(columns="_ratio_order")

    return table.reset_index(drop=True), {
        "status": "ok",
        "reference_id": requested_reference_id,
        "matched_reference_id": matched_reference_id,
        "reference_label": reference_label,
        "reference_tex_label": reference_tex_label,
        "reference_unit": ref_row.get("unit", ""),
        "reference_value_arcf4_norm": arcf4_ref.value,
        "reference_value_arn2_norm": arn2_ref.value,
    }


def write_absolute_prediction_latex(
    table: pd.DataFrame,
    tex_path: str | Path,
    config: AbsolutePredictionTableConfig,
    *,
    arcf4_label: str = r"Norm. Ar--CF$_4$",
    arn2_label: str = r"Norm. Ar--N$_2$",
) -> None:
    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{config.caption}}}",
        rf"\label{{{config.label}}}",
    ]

    if config.include_ratio_column:
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(rf"Caso & Razón integral & {arcf4_label} & {arn2_label} \\")
    else:
        lines.append(r"\begin{tabular}{lcc}")
        lines.append(r"\toprule")
        lines.append(rf"Caso & {arcf4_label} & {arn2_label} \\")
    lines.append(r"\midrule")

    for _, row in table.iterrows():
        case = _escape_latex_text(str(row["case"]))
        arcf4 = _format_value_with_total_uncertainty(
            row["value_arcf4_norm"],
            row["total_minus_arcf4_norm"],
            row["total_plus_arcf4_norm"],
            digits=config.digits,
        )
        arn2 = _format_value_with_total_uncertainty(
            row["value_arn2_norm"],
            row["total_minus_arn2_norm"],
            row["total_plus_arn2_norm"],
            digits=config.digits,
        )
        if config.include_ratio_column:
            ratio = _format_num(row["ratio"], digits=config.digits)
            lines.append(rf"{case} & {ratio} & {arcf4} & {arn2} \\")
        else:
            lines.append(rf"{case} & {arcf4} & {arn2} \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\par\vspace{{0.35em}}\footnotesize{{Valores en {config.unit}; incertidumbre total propagada solo desde la predicción primaria de referencia. El factor integral se trata como central.}}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_anchor_ratio(
    ratio_summary: pd.DataFrame,
    reference: PrimaryPredictionReference,
    ratio_titles: Mapping[str, str],
) -> float | None:
    if reference.scale_mode == "direct_ratio":
        return None
    if reference.scale_mode != "relative_to_anchor":
        raise ValueError(f"Unknown scale_mode: {reference.scale_mode!r}")

    if ratio_summary.empty:
        raise ValueError("Cannot resolve anchor ratio from an empty summary table.")

    if reference.anchor_ratio_name:
        names = ratio_summary.get("ratio_name", pd.Series("", index=ratio_summary.index, dtype=str)).astype(str)
        match = ratio_summary.loc[names.eq(str(reference.anchor_ratio_name))].copy()
    else:
        match = ratio_summary.iloc[0:0].copy()

    if match.empty:
        labels = []
        for _, row in ratio_summary.iterrows():
            ratio_name = str(row.get("ratio_name", row.get("case", "")))
            labels.append(ratio_titles.get(ratio_name, _default_case_label(row)))
        labels_s = pd.Series(labels, index=ratio_summary.index, dtype=str)
        match = ratio_summary[labels_s.eq(str(reference.anchor_case_label))].copy()

    if match.empty:
        available = ", ".join(ratio_summary.get("ratio_name", pd.Series("", index=ratio_summary.index, dtype=str)).astype(str).tolist())
        raise KeyError(
            f"Anchor ratio {reference.anchor_ratio_name!r} / {reference.anchor_case_label!r} not found. "
            f"Available ratio names: {available}"
        )

    anchor_ratio = float(match.iloc[0].get("ratio", np.nan))
    if not np.isfinite(anchor_ratio) or anchor_ratio == 0.0:
        raise ValueError(f"Invalid anchor ratio for {reference.anchor_ratio_name!r}: {anchor_ratio}")
    return anchor_ratio


def _scale_factor_from_ratio(ratio: float, anchor_ratio: float | None, scale_mode: str) -> float:
    if scale_mode == "direct_ratio":
        return float(ratio)
    if scale_mode == "relative_to_anchor":
        if anchor_ratio is None or anchor_ratio == 0.0 or not np.isfinite(anchor_ratio):
            return np.nan
        return float(ratio) / float(anchor_ratio)
    raise ValueError(f"Unknown scale_mode: {scale_mode!r}")


def _reference_from_wide_row(row: pd.Series, suffix: str) -> NormalizedReferenceValue:
    return NormalizedReferenceValue(
        value=float(row[f"value_{suffix}"]),
        stat_minus=float(row.get(f"stat_minus_{suffix}", 0.0)),
        stat_plus=float(row.get(f"stat_plus_{suffix}", 0.0)),
        syst_minus=float(row.get(f"syst_minus_{suffix}", 0.0)),
        syst_plus=float(row.get(f"syst_plus_{suffix}", 0.0)),
        total_minus=float(row.get(f"total_minus_{suffix}", 0.0)),
        total_plus=float(row.get(f"total_plus_{suffix}", 0.0)),
    )


def _scaled_columns(prefix: str, ratio: float, ref: NormalizedReferenceValue) -> dict[str, float]:
    scale = abs(float(ratio))
    return {
        f"value_{prefix}": ratio * ref.value,
        f"stat_minus_{prefix}": scale * ref.stat_minus,
        f"stat_plus_{prefix}": scale * ref.stat_plus,
        f"syst_minus_{prefix}": scale * ref.syst_minus,
        f"syst_plus_{prefix}": scale * ref.syst_plus,
        f"total_minus_{prefix}": scale * ref.total_minus,
        f"total_plus_{prefix}": scale * ref.total_plus,
    }


def _default_case_label(row: pd.Series) -> str:
    spectrum_column = str(row.get("spectrum_column", row.get("numerator_spectrum_column", "")))
    method = str(row.get("method", row.get("numerator_method", "")))
    spectrum_label = spectrum_column.replace("_spectrum", "") or "spectrum"
    method_label = {
        "trapz": "hardcut",
        "gaussian_fit": "gaussian",
    }.get(method, method or "method")
    return f"{spectrum_label}, {method_label}"


def _format_value_with_total_uncertainty(value: object, minus: object, plus: object, *, digits: int) -> str:
    v = float(value)
    m = float(minus)
    p = float(plus)
    if not np.isfinite(v):
        return r"--"
    if (not np.isfinite(m) or m == 0.0) and (not np.isfinite(p) or p == 0.0):
        return _format_num(v, digits=digits)
    return rf"{_format_num(v, digits=digits)}$^{{+{_format_num(p, digits=digits)}}}_{{-{_format_num(m, digits=digits)}}}$"


def _format_num(value: object, *, digits: int) -> str:
    x = float(value)
    if not np.isfinite(x):
        return r"--"
    return rf"\num{{{x:.{digits}g}}}"


def _escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)
