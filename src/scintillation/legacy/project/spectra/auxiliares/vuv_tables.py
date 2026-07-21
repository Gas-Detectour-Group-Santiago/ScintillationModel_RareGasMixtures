from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from spectra import config as cfg
from .common import ensure_parent, read_parameter_vector, require_file


def _add_models_to_path(project_root: Path) -> None:
    models_path = str(Path(project_root) / "models")
    if models_path not in sys.path:
        sys.path.insert(0, models_path)


def _fmt_num(value: float, precision: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "--"
    return rf"\num{{{float(value):.{precision}g}}}"


def _fmt_percent(value: float, precision: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "--"
    return rf"\num{{{float(value):.{precision}g}}}\%"


def _write_latex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    tabular: str,
    header_lines: Iterable[str],
    body_lines: Iterable[str],
    small: bool = False,
) -> None:
    ensure_parent(path)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
    ]
    if small:
        lines.append(r"\small")
    lines.extend(
        [
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{tabular}}}",
            r"\toprule",
            *header_lines,
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _energy_for_gas(gas: str) -> float:
    if gas == "ArCF4":
        from ArCF4 import energy_X_ray_CF4

        return float(energy_X_ray_CF4)
    if gas == "ArN2":
        from ArN2 import energy_X_ray_N2

        return float(energy_X_ray_N2)
    raise ValueError(f"Gas no soportado para energia X-ray: {gas}")


def _singlet_fraction(params: dict[str, float], gas: str, f_additive: float | None = None) -> float:
    """Singlet fraction of the common Ar2* branch used by the TFM model."""
    del gas, f_additive
    value = params.get("f_1Sigma", params.get("f_S", 0.1))
    return float(np.clip(float(value), 0.0, 1.0))


def _degrad_path(project_root: Path, gas: str) -> Path:
    ar2nd_paths = getattr(cfg, "AR2ND_DEGRAD_CSVS", {})
    return require_file(project_root / ar2nd_paths.get(gas, cfg.GASES[gas].degrad_csv))


def _gaussian_peak_from_integral(integral: float, sigma_nm: float) -> float:
    sigma_nm = max(float(sigma_nm), 1.0e-30)
    return float(integral) / (sigma_nm * np.sqrt(2.0 * np.pi))


def _ar2nd_prediction_row(
    *,
    row_id: str,
    gas_mosaic: str,
    degrad_reference: str,
    gas_mixture_for_rates: str,
    params: dict[str, float],
    project_root: Path,
    concentration_fraction: float,
    pressure_bar: float,
    used_in_generated_amplied: bool,
    note: str,
) -> dict[str, object]:
    from Ar2nd_continium import _effective_ar_4s_population, theory_yield_ar2nd_continium

    degrad = pd.read_csv(_degrad_path(project_root, degrad_reference))
    f = np.asarray([float(concentration_fraction)], dtype=float)
    energy = _energy_for_gas(degrad_reference)
    fast = float(
        np.asarray(
            theory_yield_ar2nd_continium(
                params,
                degrad,
                f,
                pressure_bar,
                gas_mixture=gas_mixture_for_rates,
                energy_xray_ev=energy,
            ),
            dtype=float,
        )[0]
        * 1.0e3
    )
    total_params = dict(params)
    total_params["triplet_weight"] = 1.0
    total = float(
        np.asarray(
            theory_yield_ar2nd_continium(
                total_params,
                degrad,
                f,
                pressure_bar,
                gas_mixture=gas_mixture_for_rates,
                energy_xray_ev=energy,
            ),
            dtype=float,
        )[0]
        * 1.0e3
    )
    n_eff = float(_effective_ar_4s_population(params, degrad, f)[0])
    f_s = _singlet_fraction(params, gas_mixture_for_rates, float(concentration_fraction))
    sigma = float(params["sigma_Ar2nd_nm"])
    return {
        "id": row_id,
        "gas_mosaic": gas_mosaic,
        "degrad_reference": degrad_reference,
        "gas_mixture_for_rates": gas_mixture_for_rates,
        "energy_xray_kev": energy,
        "concentration_fraction": float(concentration_fraction),
        "pressure_bar": float(pressure_bar),
        "n_ar_2nd_precursor": n_eff,
        "f_singlet_fast": f_s,
        "f_triplet_slow": 1.0 - f_s,
        "y_ar2nd_fast_ph_MeV": fast,
        "y_ar2nd_total_ph_MeV": total,
        "peak_fast_ph_MeV_nm": _gaussian_peak_from_integral(fast, sigma),
        "peak_total_ph_MeV_nm": _gaussian_peak_from_integral(total, sigma),
        "used_in_generated_amplied": bool(used_in_generated_amplied),
        "note": note,
    }


def build_ar2nd_pure_argon_table(project_root: Path) -> pd.DataFrame:
    """Build the pure-Ar Ar second-continuum table from dedicated Ar2nd CSVs."""
    _add_models_to_path(project_root)
    from Ar2nd_continium import read_ar2nd_parameters

    params = read_ar2nd_parameters(project_root / cfg.AR2ND_CONTINIUM_PARAMETER_CSV)
    pressure = 1.0
    f_pure = float(getattr(cfg, "AR2ND_PURE_REFERENCE_CONCENTRATION_PERCENT", 0.0)) / 100.0
    force_common = bool(getattr(cfg, "AR2ND_FORCE_COMMON_PURE_REFERENCE", False))
    reference_gas = str(getattr(cfg, "AR2ND_PURE_REFERENCE_GAS", "ArN2"))

    rows = []
    for gas in ("ArCF4", "ArN2"):
        degrad_reference = reference_gas if force_common else gas
        gas_mixture_for_rates = reference_gas if force_common else gas
        rows.append(
            _ar2nd_prediction_row(
                row_id=f"generated_{gas}_pure_panel",
                gas_mosaic=gas,
                degrad_reference=degrad_reference,
                gas_mixture_for_rates=gas_mixture_for_rates,
                params=params,
                project_root=project_root,
                concentration_fraction=f_pure,
                pressure_bar=pressure,
                used_in_generated_amplied=True,
                note=(
                    "Value used in the 0% additive extended spectrum. "
                    "The dedicated Ar2nd CSV contains Ar(1s4,1s5) + Ar(1s2,1s3) + Ar**."
                ),
            )
        )

    df = pd.DataFrame(rows)
    ref_fast = float(df.loc[df["gas_mosaic"] == "ArN2", "y_ar2nd_fast_ph_MeV"].iloc[0])
    ref_total = float(df.loc[df["gas_mosaic"] == "ArN2", "y_ar2nd_total_ph_MeV"].iloc[0])
    df["relative_difference_fast_vs_ArN2_percent"] = np.where(
        ref_fast != 0.0,
        100.0 * (df["y_ar2nd_fast_ph_MeV"].astype(float) - ref_fast) / ref_fast,
        np.nan,
    )
    df["relative_difference_total_vs_ArN2_percent"] = np.where(
        ref_total != 0.0,
        100.0 * (df["y_ar2nd_total_ph_MeV"].astype(float) - ref_total) / ref_total,
        np.nan,
    )
    return df


def build_cf4_d_to_x_table(project_root: Path) -> pd.DataFrame:
    """Build the CF4+*(D)->CF4+*(X) VUV prediction table."""
    _add_models_to_path(project_root)
    from Ar2nd_continium import read_ar2nd_parameters
    from ArCF4 import theory_yield_uv

    ar2_params = read_ar2nd_parameters(project_root / cfg.AR2ND_CONTINIUM_PARAMETER_CSV)
    files = cfg.GASES["ArCF4"]
    degrad = pd.read_csv(require_file(project_root / files.degrad_csv))
    params = read_parameter_vector(project_root / files.parameter_csv)
    f = np.asarray([1.0], dtype=float)
    pressure = 1.0
    _, y_cf4_uv_per_kev, _, _ = theory_yield_uv(params, degrad, f, pressure, activate_components=True)
    y_cf4_uv_raw_ph_MeV = float(np.asarray(y_cf4_uv_per_kev, dtype=float)[0] * 1.0e3)
    br = float(ar2_params["Br_CF4_D_to_X"])
    raw_vuv = br * y_cf4_uv_raw_ph_MeV

    norm_arcf4 = float(read_parameter_vector(project_root / cfg.GASES["ArCF4"].parameter_csv)[0])
    # Do not use cfg.GASES["ArN2"].norm_parameter_csv here: in the spectra
    # configuration that entry can deliberately point to ArCF4_primary.csv for
    # common-scale plots.  This table compares the two actual fitted Nnorm values.
    norm_arn2 = float(read_parameter_vector(project_root / cfg.GASES["ArN2"].parameter_csv)[0])
    value_arcf4_norm = raw_vuv / norm_arcf4
    value_arn2_norm = raw_vuv / norm_arn2
    rel_diff_percent = 100.0 * (value_arcf4_norm - value_arn2_norm) / value_arcf4_norm
    return pd.DataFrame(
        [
            {
                "id": "CF4_D_to_X_VUV_CF4",
                "label": r"$Y_{\mathrm{CF_4^+(D\to X)}}(100\%\,\mathrm{CF_4})$",
                "concentration_fraction": 1.0,
                "pressure_bar": pressure,
                "branching_ratio_D_to_X": br,
                "cf4_ionic_uv_raw_ph_MeV_before_norm": y_cf4_uv_raw_ph_MeV,
                "value_arcf4_norm_ph_MeV": value_arcf4_norm,
                "value_arn2_norm_ph_MeV": value_arn2_norm,
                "ratio_arn2_over_arcf4_norm": value_arn2_norm / value_arcf4_norm if value_arcf4_norm != 0.0 else np.nan,
                "relative_difference_percent": rel_diff_percent,
                "relative_difference_definition": "100 * (ArCF4_norm - ArN2_norm) / ArCF4_norm",
                "note": "Br(D->X) times the fitted CF4 ionic UV component; only the external normalization is changed.",
            }
        ]
    )


def write_ar2nd_pure_argon_table(df: pd.DataFrame, table_path: Path) -> None:
    labels = {
        "ArCF4": r"Ar--CF$_4$",
        "ArN2": r"Ar--N$_2$",
    }
    body = []
    for _, row in df.iterrows():
        body.append(
            " & ".join(
                [
                    labels.get(str(row["gas_mosaic"]), str(row["gas_mosaic"])),
                    str(row["degrad_reference"]).replace("ArCF4", r"Ar--CF$_4$").replace("ArN2", r"Ar--N$_2$"),
                    _fmt_num(float(row["energy_xray_kev"]), 3),
                    _fmt_num(float(row["n_ar_2nd_precursor"]), 4),
                    rf"{_fmt_num(float(row['f_singlet_fast']), 3)} / {_fmt_num(float(row['f_triplet_slow']), 3)}",
                    _fmt_num(float(row["y_ar2nd_fast_ph_MeV"]), 4),
                    _fmt_num(float(row["y_ar2nd_total_ph_MeV"]), 4),
                    _fmt_percent(float(row["relative_difference_fast_vs_ArN2_percent"]), 3),
                    _fmt_percent(float(row["relative_difference_total_vs_ArN2_percent"]), 3),
                ]
            )
            + r" \\"  # noqa: W605
        )
    _write_latex_table(
        table_path,
        caption=(
            r"Predicción del segundo continuo de argón en el límite de argón puro. "
            r"Los nuevos CSVs dedicados usan "
            r"$N_{\mathrm{prec}}=N_{\mathrm{Ar,meta}}+N_{\mathrm{Ar,res}}+N_{\mathrm{Ar}^{**}}$. "
            r"La diferencia grande en la componente rápida procede de la partición "
            r"$f_S/f_T$ usada en cada rama; el total rápido+lento queda casi igual."
        ),
        label="tab:primary_ar2nd_continuum_pure_argon",
        tabular="lcccccccc",
        header_lines=[
            r"Caso & Ref. Degrad & $E_X$ [keV] & $N_{\mathrm{prec}}$ & $f_S/f_T$ & $Y_S$ & $Y_{\mathrm{tot}}$ & $\Delta_S$ & $\Delta_{\mathrm{tot}}$ \\",
        ],
        body_lines=body,
        small=True,
    )


def write_cf4_d_to_x_table(df: pd.DataFrame, table_path: Path) -> None:
    row = df.iloc[0]
    body = [
        " & ".join(
            [
                str(row["label"]),
                _fmt_num(float(row["value_arcf4_norm_ph_MeV"]), 3),
                _fmt_num(float(row["value_arn2_norm_ph_MeV"]), 3),
                _fmt_num(float(row["ratio_arn2_over_arcf4_norm"]), 3),
                _fmt_percent(float(row["relative_difference_percent"]), 3),
            ]
        )
        + r" \\"  # noqa: W605
    ]
    _write_latex_table(
        table_path,
        caption=(
            r"Predicción VUV de la rama CF$_4^+{}^*(D)\to$CF$_4^+(X)$ en CF$_4$ puro. "
            r"La diferencia relativa se define como "
            r"$(Y_{\mathrm{ArCF_4\ norm}}-Y_{\mathrm{ArN_2\ norm}})/Y_{\mathrm{ArCF_4\ norm}}$."
        ),
        label="tab:primary_cf4_d_to_x_vuv",
        tabular="lcccc",
        header_lines=[
            r"Predicción & Norm. Ar--CF$_4$ & Norm. Ar--N$_2$ & Ratio & Diferencia relativa \\",
        ],
        body_lines=body,
    )



def build_vuv_components_table(project_root: Path) -> pd.DataFrame:
    """Build one compact table for the VUV branches used in the extended spectra."""
    _add_models_to_path(project_root)
    from Ar2nd_continium import read_ar2nd_parameters

    ar2_params = read_ar2nd_parameters(project_root / cfg.AR2ND_CONTINIUM_PARAMETER_CSV)
    ar2nd = build_ar2nd_pure_argon_table(project_root)
    cf4 = build_cf4_d_to_x_table(project_root).iloc[0]

    rows: list[dict[str, object]] = []
    gas_labels = {
        "ArCF4": r"Ar puro (mosaico Ar--CF$_4$)",
        "ArN2": r"Ar puro (mosaico Ar--N$_2$)",
    }
    for _, row in ar2nd.iterrows():
        f_s = float(row["f_singlet_fast"])
        f_t = float(row["f_triplet_slow"])
        y_fast = float(row["y_ar2nd_fast_ph_MeV"])
        y_total = float(row["y_ar2nd_total_ph_MeV"])
        rows.append(
            {
                "id": f"Ar2nd_total_{row['gas_mosaic']}_pure_panel",
                "component_tex": r"Ar$_2^*$ segundo continuo",
                "case_tex": gas_labels.get(str(row["gas_mosaic"]), str(row["gas_mosaic"])),
                "wavelength_nm": float(ar2_params["lambda_Ar2nd_nm"]),
                "yield_used_in_spectrum_ph_MeV": y_total,
                "ar2nd_fast_ph_MeV": y_fast,
                "ar2nd_total_ph_MeV": y_total,
                "cf4_value_arcf4_norm_ph_MeV": np.nan,
                "cf4_value_arn2_norm_ph_MeV": np.nan,
                "relative_difference_percent": float(row["relative_difference_total_vs_ArN2_percent"]),
                "table_note_tex": (
                    rf"total $S+T$; $Y_S={_fmt_num(y_fast, 3)}$; "
                    rf"$f_S/f_T={_fmt_num(f_s, 2)}/{_fmt_num(f_t, 2)}$"
                ),
            }
        )

    y_cf4 = float(cf4["value_arcf4_norm_ph_MeV"])
    y_cf4_arn2_norm = float(cf4["value_arn2_norm_ph_MeV"])
    rel_diff = float(cf4["relative_difference_percent"])
    rows.append(
        {
            "id": "CF4_D_to_X_VUV_CF4",
            "component_tex": r"CF$_4^+{}^*(D)\to$CF$_4^+(X)",
            "case_tex": r"CF$_4$ puro",
            "wavelength_nm": float(ar2_params["lambda_CF4_D_to_X_nm"]),
            "yield_used_in_spectrum_ph_MeV": y_cf4,
            "ar2nd_fast_ph_MeV": np.nan,
            "ar2nd_total_ph_MeV": np.nan,
            "cf4_value_arcf4_norm_ph_MeV": y_cf4,
            "cf4_value_arn2_norm_ph_MeV": y_cf4_arn2_norm,
            "relative_difference_percent": rel_diff,
            "table_note_tex": (
                rf"Br={_fmt_num(float(cf4['branching_ratio_D_to_X']), 2)}; "
                rf"norm. Ar--N$_2$: {_fmt_num(y_cf4_arn2_norm, 3)}; "
                rf"$\Delta={_fmt_percent(rel_diff, 2)}$"
            ),
        }
    )
    return pd.DataFrame(rows)


def write_vuv_components_table(df: pd.DataFrame, table_path: Path) -> None:
    """Write one space-saving LaTeX table for the two VUV components."""
    body = []
    for _, row in df.iterrows():
        body.append(
            " & ".join(
                [
                    str(row["component_tex"]),
                    str(row["case_tex"]),
                    _fmt_num(float(row["wavelength_nm"]), 3),
                    _fmt_num(float(row["yield_used_in_spectrum_ph_MeV"]), 3),
                    str(row["table_note_tex"]),
                ]
            )
            + r" \\"  # noqa: W605
        )

    _write_latex_table(
        table_path,
        caption=(
            r"Predicciones de las componentes VUV añadidas a los espectros extendidos. "
            r"Para el segundo continuo del argón se muestra el rendimiento total "
            r"$S+T$, que es el usado en la representación espectral no resuelta en tiempo."
        ),
        label="tab:primary_vuv_components",
        tabular=r"llccp{0.34\textwidth}",
        header_lines=[
            r"Componente & Caso & $\lambda$ [nm] & $Y_{\mathrm{VUV}}$ [ph/MeV] & Nota \\",
        ],
        body_lines=body,
        small=True,
    )

def export_vuv_prediction_tables(project_root: Path) -> dict[str, Path]:
    """Export one compact CSV and LaTeX table for the extended VUV branches."""
    project_root = Path(project_root)
    pred_dir = project_root / "data" / "Predictions"
    table_dir = project_root / "data" / "Tables"
    pred_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    vuv = build_vuv_components_table(project_root)
    vuv_csv = pred_dir / "primary_vuv_components.csv"
    vuv_tex = table_dir / "primary_vuv_components.tex"
    vuv.to_csv(vuv_csv, index=False)
    write_vuv_components_table(vuv, vuv_tex)

    # Avoid keeping stale split tables after switching to the compact output.
    for stale in (
        pred_dir / "primary_ar2nd_continuum_pure_argon.csv",
        pred_dir / "primary_cf4_d_to_x_vuv.csv",
        table_dir / "primary_ar2nd_continuum_pure_argon.tex",
        table_dir / "primary_cf4_d_to_x_vuv.tex",
    ):
        if stale.exists():
            stale.unlink()

    print(f"[spectra] VUV prediction CSV: {vuv_csv}")
    print(f"[spectra] VUV prediction table: {vuv_tex}")
    return {
        "vuv_csv": vuv_csv,
        "vuv_tex": vuv_tex,
    }
