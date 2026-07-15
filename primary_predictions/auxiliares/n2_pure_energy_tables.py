from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .bands import asymmetric_errors
from .fit_products import FitProductStore
from .model_adapters import apply_normalization, prepare_parameters
from .prediction_types import NormalizationConfig
from .tables import fmt_asym, fmt_num


N2_XRAY_MODEL_ENERGY_KEV = 12.0


@dataclass(frozen=True)
class N2PureEnergyCase:
    id: str
    tex_label: str
    particle: str
    energy_kev: float
    electric_field_v_cm: float


DEFAULT_CASE_ORDER: tuple[str, ...] = (
    "N2_pure_xray_12keV",
    "N2_pure_electron_1p5MeV_0Vcm",
    "N2_pure_electron_1p5MeV_50Vcm",
)


def build_n2_pure_energy_prediction_table(
    project_root: Path,
    *,
    left_normalization: NormalizationConfig,
    right_normalization: NormalizationConfig,
    left_prefix: str = "arcf4_norm",
    right_prefix: str = "arn2_norm",
    fit_name: str = "ArN2_primary",
    population_csv: Path | None = None,
) -> pd.DataFrame:
    """Predict pure-N2 UV yield for the 12 keV and 1.5 MeV Degrad cases.

    The Ar--N2 primary model was fitted with 12 keV X-ray Degrad populations and
    divides the microscopic population by 12 keV internally.  For a dedicated
    1.5 MeV electron Degrad run, the same kinetic probabilities are evaluated
    but the raw model is rescaled by 12/E so the output remains ph/MeV after the
    usual primary-prediction normalisation.
    """

    project_root = Path(project_root)
    population_csv = population_csv or (project_root / "data" / "Primary_DegradData" / "ArN2_pure_energy_cases.csv")
    if not population_csv.exists():
        _try_generate_population_csv(project_root)
    if not population_csv.exists():
        raise FileNotFoundError(
            f"No encuentro {population_csv}. Ejecuta data/Analysis_primary_degrad.py "
            "después de mover los Outupt_1.5MeV_*.txt a data/Primary_DegradData/ArN2/txt."
        )

    cases = pd.read_csv(population_csv)
    if cases.empty:
        raise ValueError(f"{population_csv} está vacío")
    cases = _order_cases(cases)

    store = FitProductStore(project_root)
    product = store.load(fit_name)

    rows: list[dict[str, object]] = []
    for _, case in cases.iterrows():
        base_row = {
            "id": str(case["id"]),
            "tex_label": str(case.get("tex_label", case["id"])),
            "particle": str(case.get("particle", "")),
            "energy_kev": float(case["energy_kev"]),
            "electric_field_v_cm": float(case.get("electric_field_v_cm", np.nan)),
            "concentration": float(case.get("concentration", 1.0)),
            "pressure_bar": float(case.get("pressure_bar", 1.0)),
            "unit": left_normalization.output_unit,
        }
        row_df = pd.DataFrame([case])
        base_row.update(
            _evaluate_case_with_normalization(
                product.central,
                product.stat_toys,
                product.syst_toys,
                row_df,
                energy_kev=float(case["energy_kev"]),
                pressure_bar=float(case.get("pressure_bar", 1.0)),
                normalization=left_normalization,
                reference_norm=float(store.load(left_normalization.reference_fit_name or fit_name).central[0])
                if left_normalization.mode == "reference_norm"
                else None,
                central_params=product.central,
                prefix=left_prefix,
            )
        )
        base_row.update(
            _evaluate_case_with_normalization(
                product.central,
                product.stat_toys,
                product.syst_toys,
                row_df,
                energy_kev=float(case["energy_kev"]),
                pressure_bar=float(case.get("pressure_bar", 1.0)),
                normalization=right_normalization,
                reference_norm=float(store.load(right_normalization.reference_fit_name or fit_name).central[0])
                if right_normalization.mode == "reference_norm"
                else None,
                central_params=product.central,
                prefix=right_prefix,
            )
        )
        rows.append(base_row)

    return pd.DataFrame(rows)


def write_n2_pure_energy_prediction_latex(
    df: pd.DataFrame,
    tex_path: str | Path,
    *,
    caption: str,
    label: str,
    left_prefix: str = "arcf4_norm",
    right_prefix: str = "arn2_norm",
    left_heading: str = r"Norm. Ar--CF$_4$",
    right_heading: str = r"Norm. Ar--N$_2$",
) -> None:
    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

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
        r"Entrada Degrad & Valor & Stat. & Syst. & Valor & Stat. & Syst. \\",
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
            f"{fmt_asym(row[col('syst_minus', right_prefix)], row[col('syst_plus', right_prefix)])} " r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def export_n2_pure_energy_prediction_table(
    project_root: Path,
    stem: str,
    *,
    caption: str,
    label: str,
    left_normalization: NormalizationConfig,
    right_normalization: NormalizationConfig,
) -> pd.DataFrame:
    project_root = Path(project_root)
    df = build_n2_pure_energy_prediction_table(
        project_root,
        left_normalization=left_normalization,
        right_normalization=right_normalization,
    )
    csv_path = project_root / "data" / "Predictions" / f"{stem}.csv"
    tex_path = project_root / "data" / "Tables" / f"{stem}.tex"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_n2_pure_energy_prediction_latex(df, tex_path, caption=caption, label=label)
    print(f"[primary_predictions] tabla N2 puro por entrada CSV: {csv_path}")
    print(f"[primary_predictions] tabla N2 puro por entrada TeX: {tex_path}")
    return df


def _evaluate_case_with_normalization(
    central: np.ndarray,
    stat_toys: np.ndarray,
    syst_toys: np.ndarray,
    degrad_row: pd.DataFrame,
    *,
    energy_kev: float,
    pressure_bar: float,
    normalization: NormalizationConfig,
    reference_norm: float | None,
    central_params: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    central_value = float(
        np.ravel(
            _evaluate_n2_case(
                central,
                degrad_row,
                energy_kev=energy_kev,
                pressure_bar=pressure_bar,
                normalization=normalization,
                reference_norm=reference_norm,
                central_params=central_params,
            )
        )[0]
    )
    stat_samples = _evaluate_samples(
        stat_toys,
        degrad_row,
        energy_kev=energy_kev,
        pressure_bar=pressure_bar,
        normalization=normalization,
        reference_norm=reference_norm,
        central_params=central_params,
    )
    syst_samples = _evaluate_samples(
        syst_toys,
        degrad_row,
        energy_kev=energy_kev,
        pressure_bar=pressure_bar,
        normalization=normalization,
        reference_norm=reference_norm,
        central_params=central_params,
    )
    stat_minus, stat_plus = asymmetric_errors(
        np.asarray([central_value]),
        stat_samples[:, None] if stat_samples.size else np.empty((0, 1)),
    )
    syst_minus, syst_plus = asymmetric_errors(
        np.asarray([central_value]),
        syst_samples[:, None] if syst_samples.size else np.empty((0, 1)),
    )
    sm = float(stat_minus[0]) if len(stat_minus) else np.nan
    sp = float(stat_plus[0]) if len(stat_plus) else np.nan
    ym = float(syst_minus[0]) if len(syst_minus) else np.nan
    yp = float(syst_plus[0]) if len(syst_plus) else np.nan
    return {
        f"value_{prefix}": central_value,
        f"stat_minus_{prefix}": sm,
        f"stat_plus_{prefix}": sp,
        f"syst_minus_{prefix}": ym,
        f"syst_plus_{prefix}": yp,
        f"total_minus_{prefix}": float(np.sqrt(np.nan_to_num(sm) ** 2 + np.nan_to_num(ym) ** 2)),
        f"total_plus_{prefix}": float(np.sqrt(np.nan_to_num(sp) ** 2 + np.nan_to_num(yp) ** 2)),
    }


def _evaluate_samples(
    toys: np.ndarray,
    degrad_row: pd.DataFrame,
    *,
    energy_kev: float,
    pressure_bar: float,
    normalization: NormalizationConfig,
    reference_norm: float | None,
    central_params: np.ndarray,
) -> np.ndarray:
    if toys.ndim != 2 or toys.shape[0] == 0:
        return np.empty((0,), dtype=float)
    values: list[float] = []
    for params in toys:
        try:
            values.append(
                float(
                    np.ravel(
                        _evaluate_n2_case(
                            params,
                            degrad_row,
                            energy_kev=energy_kev,
                            pressure_bar=pressure_bar,
                            normalization=normalization,
                            reference_norm=reference_norm,
                            central_params=central_params,
                        )
                    )[0]
                )
            )
        except Exception:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _evaluate_n2_case(
    params: np.ndarray,
    degrad_row: pd.DataFrame,
    *,
    energy_kev: float,
    pressure_bar: float,
    normalization: NormalizationConfig,
    reference_norm: float | None,
    central_params: np.ndarray,
) -> np.ndarray:
    """Evaluate the Ar--N2 UV model for a single pure-N2 Degrad row.

    ``models.ArN2.theory_yield_N2_uv`` interpolates over concentration and
    therefore requires at least two scan points.  These dedicated files are
    single pure-N2 rows, so here we evaluate the same pure-N2 limit explicitly.
    """

    params_eval = prepare_parameters(params, normalization, central_params=central_params)
    pob_n2 = float(pd.to_numeric(degrad_row.iloc[0].get("N2_star", 0.0), errors="coerce"))

    n_norm = float(params_eval[0])
    p_n2 = float(params_eval[1])
    tau_n2 = max(float(params_eval[2]), 1.0e-30)
    k_n2_q_n2 = max(float(params_eval[3]), 0.0)

    n = float(pressure_bar)
    inv_tau = 1.0 / tau_n2
    factor_n2 = inv_tau / (inv_tau + n * k_n2_q_n2)

    raw = np.asarray([n_norm * factor_n2 * pob_n2 * p_n2 / float(energy_kev)], dtype=float)
    return apply_normalization(raw, params_eval, normalization, reference_norm=reference_norm)


def _order_cases(df: pd.DataFrame) -> pd.DataFrame:
    order = {case_id: idx for idx, case_id in enumerate(DEFAULT_CASE_ORDER)}
    out = df.copy()
    out["_order"] = out["id"].astype(str).map(order).fillna(len(order)).astype(int)
    return out.sort_values(["_order", "id"]).drop(columns="_order").reset_index(drop=True)


def _try_generate_population_csv(project_root: Path) -> None:
    try:
        import sys

        data_dir = project_root / "data"
        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))
        from Analysis_primary_degrad import analyse_n2_pure_energy_cases

        analyse_n2_pure_energy_cases()
    except Exception as exc:  # pragma: no cover - only a convenience fallback
        print(f"[primary_predictions] no pude generar ArN2_pure_energy_cases.csv automáticamente: {exc}")
