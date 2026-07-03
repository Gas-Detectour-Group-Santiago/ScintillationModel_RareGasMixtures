from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


AR_1S_CANDIDATES: tuple[str, ...] = (
    "Ar_1s",
    "Ar_4s",
    "Ar_meta",
    "Ar_res",
    "Ar_star_1s",
    "Ar_star_4s",
)
AR_UPPER_CANDIDATES: tuple[str, ...] = (
    "Ar_dbleStar",
    "Ar_dblestar",
    "Ar_doubleStar",
    "Ar_dblStar",
    "Ar_upper",
)
AR_2ND_PRECURSOR_CANDIDATES: tuple[str, ...] = (
    "Ar_2nd_precursor",
    "Ar2nd_precursor",
    "Ar_second_continuum_precursor",
)


DEFAULT_PARAMETERS: dict[str, float] = {
    # Overall optical weight from upper Ar states into the 4s/1s precursor family.
    "W_Ar_dbleStar_to_1s": 1.0,
    # Optional global scale. Keep at 1 unless the VUV branch is externally calibrated.
    "scale_Ar2nd": 1.0,
    # Keep the direct kinetic output by default; do not force the band to 2e4 ph/MeV.
    "anchor_Ar2nd_to_pure_argon": 0.0,
    "Y_Ar2nd_pure_ph_MeV": 2.0e4,
    "reference_pressure_bar": 1.0,
    "reference_additive_fraction": 1.0e-5,
    # Kinetic competition for Ar(4s/1s) -> Ar2* formation. Units follow the rest
    # of the project convention: pressure-normalised rates in ns^-1 at n = p / 1 bar.
    "K_Ar_star_Q_2Ar": 1.586e-2,
    "K_Ar_star_Q_Ar": 0.0,
    "K_Ar_star_Q_CF4": 0.0,
    # Effective N2 competition rate from the existing Ar--N2 primary model:
    # K_ArMeta_Q_N2c + K_ArMeta_Q_N2b.
    "K_Ar_star_Q_N2": 8.43085976942e-1,
    # Finite effective lifetime/loss of the Ar(4s/1s) precursor.  Keeping it
    # infinite makes P_form saturated and removes the pressure dependence.
    "tau_Ar_star_ns": 30.0,
    # Fast/slow excimer parameters. f_S corresponds to NT/NS = 5.5.
    # The default prediction keeps only the prompt singlet contribution for
    # photon-feedback studies; the triplet can be restored with triplet_weight=1.
    "tau_S_ns": 4.2,
    "tau_T_ns": 3200.0,
    "f_S": 1.0 / 6.5,
    # Gas-dependent singlet/fast fractions used by the second-continuum branch.
    # ``f_S`` above remains as the backwards-compatible fallback.
    "f_S_CF4": 0.154,
    "f_S_ArCF4": 0.154,
    "f_S_N2": 0.635,
    "f_S_ArN2": 0.635,
    "triplet_weight": 0.0,
    # Excimer quenching. Unknown additive channels can be left at zero, giving an
    # upper-limit prediction as described in the text.
    "K_Ar2starS_Q_Ar": 0.0,
    "K_Ar2starT_Q_Ar": 0.0,
    "K_Ar2starS_Q_CF4": 0.0,
    "K_Ar2starT_Q_CF4": 0.0,
    "K_Ar2starS_Q_N2": 0.0,
    # Effective triplet quenching by N2, kept configurable.  The singlet channel
    # is left at zero because the slow component is the one most affected.
    "K_Ar2starT_Q_N2": 1.0e-1,
    # Shape of the Ar second-continuum band.  The CSV exposes FWHM because that
    # is the usual experimental width; sigma is computed at read time.
    "lambda_Ar2nd_nm": 128.0,
    "fwhm_Ar2nd_nm": 10.0,
    "sigma_Ar2nd_nm": 10.0 / 2.354820045,
    # Phenomenological CF4+*(D)->CF4+(X) VUV branch around 150--155 nm.
    "Br_CF4_D_to_X": 0.1,
    "lambda_CF4_D_to_X_nm": 155.0,
    "fwhm_CF4_D_to_X_nm": 10.0,
    "sigma_CF4_D_to_X_nm": 10.0 / 2.354820045,
}


def read_ar2nd_parameters(path: str | Path) -> dict[str, float]:
    """Read the Ar second-continuum parameter CSV.

    The file is intentionally name-based, so parameters can be reordered without
    breaking the kinetic model. Missing parameters fall back to DEFAULT_PARAMETERS.
    """
    params = dict(DEFAULT_PARAMETERS)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el CSV de parámetros del segundo continuo: {path}")
    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError(f"{path} debe contener una columna 'name'")
    value_col = "value" if "value" in df.columns else None
    if value_col is None:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            raise ValueError(f"{path} debe contener una columna numérica de valores")
        value_col = str(numeric.columns[0])
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        if not name:
            continue
        value = row[value_col]
        if pd.isna(value):
            continue
        params[name] = float(value)
    return _finalise_parameters(params)


def _finalise_parameters(params: dict[str, float]) -> dict[str, float]:
    """Derive Gaussian sigma values from FWHM when present in the CSV."""
    fwhm_to_sigma = 1.0 / 2.354820045
    if "fwhm_Ar2nd_nm" in params:
        params["sigma_Ar2nd_nm"] = max(float(params["fwhm_Ar2nd_nm"]), 1.0e-12) * fwhm_to_sigma
    if "fwhm_CF4_D_to_X_nm" in params:
        params["sigma_CF4_D_to_X_nm"] = max(float(params["fwhm_CF4_D_to_X_nm"]), 1.0e-12) * fwhm_to_sigma
    return params


def _parameter_dict(params: Mapping[str, float] | pd.DataFrame | None) -> dict[str, float]:
    if params is None:
        return _finalise_parameters(dict(DEFAULT_PARAMETERS))
    if isinstance(params, pd.DataFrame):
        out = dict(DEFAULT_PARAMETERS)
        if "name" not in params.columns:
            raise ValueError("El DataFrame de parámetros debe contener una columna 'name'")
        value_col = "value" if "value" in params.columns else params.select_dtypes(include=["number"]).columns[0]
        for _, row in params.iterrows():
            value = row[value_col]
            if pd.notna(value):
                out[str(row["name"]).strip()] = float(value)
        return _finalise_parameters(out)
    out = dict(DEFAULT_PARAMETERS)
    out.update({str(k): float(v) for k, v in params.items()})
    return _finalise_parameters(out)


def _prepare_fraction(f_additive):
    scalar_input = np.isscalar(f_additive) or np.asarray(f_additive).ndim == 0
    f = np.atleast_1d(np.asarray(f_additive, dtype=float))
    return f, scalar_input


def _interp_column(degrad_data: pd.DataFrame, f_additive: np.ndarray, candidates: tuple[str, ...]) -> np.ndarray:
    existing = [col for col in candidates if col in degrad_data.columns]
    if not existing:
        return np.zeros_like(f_additive, dtype=float)

    concentration = np.asarray(degrad_data["concentration"], dtype=float)
    values = np.zeros(len(degrad_data), dtype=float)
    for col in existing:
        values += np.asarray(degrad_data[col], dtype=float)

    idx = np.argsort(concentration)
    conc_sorted = concentration[idx]
    values_sorted = values[idx]
    conc_unique, unique_idx = np.unique(conc_sorted, return_index=True)
    values_unique = values_sorted[unique_idx]

    if len(conc_unique) == 1:
        return np.repeat(values_unique[0], len(f_additive))
    interp = PchipInterpolator(conc_unique, values_unique, extrapolate=True)
    return np.asarray(interp(f_additive), dtype=float)


def _positive_rate(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return max(value, 0.0)


def _effective_ar_4s_population(
    params: Mapping[str, float],
    degrad_data: pd.DataFrame,
    f_additive: np.ndarray,
) -> np.ndarray:
    """Effective Ar precursor population for Ar2* formation.

    New ``*_Ar2nd.csv`` files carry an explicit ``Ar_2nd_precursor`` column
    equal to Ar_meta + Ar_res + Ar_dbleStar.  When present, use it directly to
    avoid double counting.  Older tables remain supported through the historical
    fallback N_1s + W_Ar**->1s N_Ar**.
    """
    if any(col in degrad_data.columns for col in AR_2ND_PRECURSOR_CANDIDATES):
        return np.clip(_interp_column(degrad_data, f_additive, AR_2ND_PRECURSOR_CANDIDATES), 0.0, None)

    n_1s = _interp_column(degrad_data, f_additive, AR_1S_CANDIDATES)
    n_upper = _interp_column(degrad_data, f_additive, AR_UPPER_CANDIDATES)
    return np.clip(n_1s + n_upper * params["W_Ar_dbleStar_to_1s"], 0.0, None)


def _singlet_fraction(params: Mapping[str, float], gas_mixture: str) -> float:
    gas_key = gas_mixture.upper().replace("-", "")
    if "CF4" in gas_key:
        value = params.get("f_S_CF4", params.get("f_S_ArCF4", params.get("f_S", 1.0 / 6.5)))
    elif "N2" in gas_key:
        value = params.get("f_S_N2", params.get("f_S_ArN2", params.get("f_S", 1.0 / 6.5)))
    else:
        value = params.get("f_S", 1.0 / 6.5)
    return float(np.clip(float(value), 0.0, 1.0))


def _additive_rates(params: Mapping[str, float], gas_mixture: str) -> tuple[float, float, float]:
    gas_key = gas_mixture.upper().replace("-", "")
    if "CF4" in gas_key:
        return (
            _positive_rate(params["K_Ar_star_Q_CF4"]),
            _positive_rate(params["K_Ar2starS_Q_CF4"]),
            _positive_rate(params["K_Ar2starT_Q_CF4"]),
        )
    if "N2" in gas_key:
        return (
            _positive_rate(params["K_Ar_star_Q_N2"]),
            _positive_rate(params["K_Ar2starS_Q_N2"]),
            _positive_rate(params["K_Ar2starT_Q_N2"]),
        )
    return 0.0, 0.0, 0.0


def _formation_probability(params: Mapping[str, float], f_additive: np.ndarray, n: float, k_add: float) -> np.ndarray:
    f_ar = np.clip(1.0 - f_additive, 0.0, 1.0)
    k_form = _positive_rate(params["K_Ar_star_Q_2Ar"])
    k_ar = _positive_rate(params["K_Ar_star_Q_Ar"])
    tau_ar_star = max(float(params["tau_Ar_star_ns"]), 1.0e-30)
    inv_tau_ar_star = 1.0 / tau_ar_star

    form_num = (n**2) * (f_ar**2) * k_form
    form_den = form_num + n * f_ar * k_ar + n * f_additive * k_add + inv_tau_ar_star
    return np.divide(form_num, form_den, out=np.zeros_like(form_num), where=form_den > 0.0)


def _radiative_survival(
    params: Mapping[str, float],
    f_additive: np.ndarray,
    n: float,
    k_s_add: float,
    k_t_add: float,
) -> tuple[np.ndarray, np.ndarray]:
    f_ar = np.clip(1.0 - f_additive, 0.0, 1.0)
    tau_s = max(float(params["tau_S_ns"]), 1.0e-30)
    tau_t = max(float(params["tau_T_ns"]), 1.0e-30)
    inv_tau_s = 1.0 / tau_s
    inv_tau_t = 1.0 / tau_t

    k_s_ar = _positive_rate(params["K_Ar2starS_Q_Ar"])
    k_t_ar = _positive_rate(params["K_Ar2starT_Q_Ar"])

    den_s = inv_tau_s + n * f_ar * k_s_ar + n * f_additive * k_s_add
    den_t = inv_tau_t + n * f_ar * k_t_ar + n * f_additive * k_t_add
    p_rad_s = np.divide(inv_tau_s, den_s, out=np.zeros_like(f_additive), where=den_s > 0.0)
    p_rad_t = np.divide(inv_tau_t, den_t, out=np.zeros_like(f_additive), where=den_t > 0.0)
    return p_rad_s, p_rad_t


def _base_yield_per_keV(
    params: Mapping[str, float],
    degrad_data: pd.DataFrame,
    f_additive: np.ndarray,
    n: float,
    *,
    gas_mixture: str,
    energy_xray_kev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Singlet/triplet second-continuum yields per keV.

    This branch is not tied to the fitted primary optical normalisation Nnorm.
    The absolute scale comes from the degradation populations, kinetic
    probabilities, and the X-ray energy used to express the result per unit
    deposited energy.
    """
    k_add, k_s_add, k_t_add = _additive_rates(params, gas_mixture)
    n_eff = _effective_ar_4s_population(params, degrad_data, f_additive)
    p_form = _formation_probability(params, f_additive, n, k_add)
    p_rad_s, p_rad_t = _radiative_survival(params, f_additive, n, k_s_add, k_t_add)

    f_s = _singlet_fraction(params, gas_mixture)
    f_t = 1.0 - f_s
    energy_xray_kev = max(float(energy_xray_kev), 1.0e-30)

    singlet = n_eff * p_form * f_s * p_rad_s / energy_xray_kev
    triplet = n_eff * p_form * f_t * p_rad_t / energy_xray_kev
    return singlet, triplet


def theory_yield_ar2nd_continium(
    params: Mapping[str, float] | pd.DataFrame | None,
    degrad_data: pd.DataFrame,
    f_additive,
    n: float,
    *,
    gas_mixture: str,
    n_norm: float = 1.0,
    energy_xray_ev: float = 1.0,
    activate_components: bool = False,
):
    """Ar second-continuum yield using the compact model in the TFM text.

    The argument name ``energy_xray_ev`` is kept for compatibility with the
    existing primary models, but the project convention is actually keV:
    ``energy_X_ray_N2 = 12`` and ``energy_X_ray_CF4 = 15``.  The generated
    spectra then multiply by ``1e3`` to obtain ph MeV^-1 nm^-1.

    By default the Ar second-continuum channel is the direct kinetic output,
    without rescaling to a fixed pure-Ar reference.  If ``triplet_weight=0`` the
    returned component is prompt/singlet only; if ``triplet_weight=1`` it is the
    singlet+triplet total.  Internally the returned value stays in the same
    X-ray-energy convention as the existing primary models: the generated builder
    applies only the keV-to-MeV factor, without any division by a primary Nnorm.
    """
    p = _parameter_dict(params)
    f, scalar_input = _prepare_fraction(f_additive)
    n = float(n)
    energy_xray_kev = float(energy_xray_ev)

    singlet_base, triplet_base = _base_yield_per_keV(
        p,
        degrad_data,
        f,
        n,
        gas_mixture=gas_mixture,
        energy_xray_kev=energy_xray_kev,
    )

    scale = float(p.get("scale_Ar2nd", 1.0))
    triplet_weight = np.clip(float(p.get("triplet_weight", 0.0)), 0.0, 1.0)

    singlet = singlet_base * scale
    triplet = triplet_base * scale
    total = singlet + triplet_weight * triplet

    if float(p.get("anchor_Ar2nd_to_pure_argon", 0.0)) > 0.5:
        # Anchor the reported Ar second-continuum channel to a single pure-Ar
        # absolute reference.  This removes artificial dependence on the
        # primary optical normalisation and also cancels differences between
        # the Ar--CF4 and Ar--N2 degradation tables in the pure-Ar limit.
        ref_f = np.asarray([float(p.get("reference_additive_fraction", 1.0e-5))], dtype=float)
        ref_pressure = float(p.get("reference_pressure_bar", 1.0))
        ref_singlet_base, ref_triplet_base = _base_yield_per_keV(
            p,
            degrad_data,
            ref_f,
            ref_pressure,
            gas_mixture=gas_mixture,
            energy_xray_kev=energy_xray_kev,
        )
        ref_singlet = ref_singlet_base * scale
        ref_triplet = ref_triplet_base * scale
        ref_total = ref_singlet + triplet_weight * ref_triplet

        # Y_Ar2nd_pure_ph_MeV is the total pure-Ar reference.  The reported
        # component can deliberately be prompt-only, so anchor to the same
        # singlet/triplet fraction selected by triplet_weight.
        f_s_anchor = _singlet_fraction(p, gas_mixture)
        component_fraction = f_s_anchor + triplet_weight * (1.0 - f_s_anchor)
        target_ph_mev = float(p.get("Y_Ar2nd_pure_ph_MeV", 2.0e4)) * component_fraction
        target_raw = target_ph_mev / 1.0e3
        ref_value = float(np.ravel(ref_total)[0]) if np.size(ref_total) else 0.0
        if np.isfinite(ref_value) and ref_value > 0.0:
            anchor_factor = target_raw / ref_value
            singlet = singlet * anchor_factor
            triplet = triplet * anchor_factor
            total = total * anchor_factor

    if activate_components:
        if scalar_input:
            return total.item(), singlet.item(), triplet.item()
        return total, singlet, triplet
    if scalar_input:
        return total.item()
    return total
