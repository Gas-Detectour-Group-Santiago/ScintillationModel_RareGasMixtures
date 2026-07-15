from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# Loschmidt number used in the TFM (ideal gas at the reference state).
LOSCHMIDT_M3 = 2.6868e25
SECONDS_PER_NANOSECOND = 1.0e-9


AR_UPPER_CANDIDATES: tuple[str, ...] = (
    "Ar_dbleStar",
    "Ar_dblestar",
    "Ar_doubleStar",
    "Ar_dblStar",
    "Ar_upper",
)
AR_HIGH_4S_CANDIDATES: tuple[str, ...] = (
    "Ar_1s2_1s3",
    "Ar_4s_upper",
    "Ar_high_4s",
)
AR_EXCIMER_PRECURSOR_CANDIDATES: tuple[str, ...] = (
    "Ar_1s4_1s5",
    "Ar_excimer_precursor",
    "Ar_4s_precursor",
)
AR_2ND_PRECURSOR_CANDIDATES: tuple[str, ...] = (
    "Ar_2nd_precursor",
    "Ar2nd_precursor",
    "Ar_second_continuum_precursor",
)


DEFAULT_PARAMETERS: dict[str, float] = {
    # Reference density and literature coefficients from Table 19 of the TFM.
    "loschmidt_m3": LOSCHMIDT_M3,
    "W_Ar_dbleStar_to_1s": 1.0,
    "tau_Ar_dbleStar_ns": 30.0,
    "k_Ar_dbleStar_Q_Ar_m3_s": 1.63e-17,
    "k_Ar_dbleStar_Q_CF4_m3_s": 1.80e-16,
    "k_Ar_dbleStar_Q_N2_m3_s": 1.58e-16,
    "k_Ar_4s_Q_CF4_m3_s": 3.00e-17,
    "k_Ar_4s_Q_N2_m3_s": 3.60e-17,
    "k_Ar_4s_Q_2Ar_m6_s": 1.00e-44,
    "k_Ar2star_Q_CF4_m3_s": 3.00e-17,
    "k_Ar2star_Q_N2_m3_s": 2.50e-17,
    "tau_S_ns": 11.3,
    "tau_T_ns": 3140.0,
    "f_1Sigma": 0.1,
    "f_3Sigma": 0.9,
    # Selector used by the existing plotting/table interface. The kinetic model
    # always computes both components; the nominal output returns fast + slow.
    "triplet_weight": 1.0,
    "scale_Ar2nd": 1.0,
    # Disabled in the nominal TFM model: the absolute prediction comes directly
    # from Degrad populations and literature kinetics, not from Nnorm or an anchor.
    "anchor_Ar2nd_to_pure_argon": 0.0,
    "Y_Ar2nd_pure_ph_MeV": 1.47e4,
    "reference_pressure_bar": 1.1,
    "reference_additive_fraction": 1.0e-5,
    "pure_argon_fraction_threshold": 0.0,
    # Spectral shapes.
    "lambda_Ar2nd_nm": 128.0,
    "fwhm_Ar2nd_nm": 10.0,
    "sigma_Ar2nd_nm": 10.0 / 2.354820045,
    "Br_CF4_D_to_X": 0.1,
    "lambda_CF4_D_to_X_nm": 155.0,
    "fwhm_CF4_D_to_X_nm": 10.0,
    "sigma_CF4_D_to_X_nm": 10.0 / 2.354820045,
}


def _two_body_to_ns_inv(k_m3_s: float, loschmidt_m3: float) -> float:
    """Convert k [m3 s-1] to the pressure-normalised rate [ns-1]."""
    return float(k_m3_s) * float(loschmidt_m3) * SECONDS_PER_NANOSECOND


def _three_body_to_ns_inv(k_m6_s: float, loschmidt_m3: float) -> float:
    """Convert k [m6 s-1] to the pressure-normalised rate [ns-1]."""
    return float(k_m6_s) * float(loschmidt_m3) ** 2 * SECONDS_PER_NANOSECOND


def _finalise_parameters(params: dict[str, float]) -> dict[str, float]:
    """Derive pressure-normalised rates and Gaussian widths.

    With n = p/(1 bar), a two-body term is n f K and a three-body term is
    n^2 f_Ar^2 K. The K values below are obtained from the SI coefficients in
    the parameter CSV using the Loschmidt number stored in that same file.
    """
    loschmidt = max(float(params.get("loschmidt_m3", LOSCHMIDT_M3)), 0.0)

    params["K_Ar_dbleStar_Q_Ar"] = _two_body_to_ns_inv(
        params["k_Ar_dbleStar_Q_Ar_m3_s"], loschmidt
    )
    params["K_Ar_dbleStar_Q_CF4"] = _two_body_to_ns_inv(
        params["k_Ar_dbleStar_Q_CF4_m3_s"], loschmidt
    )
    params["K_Ar_dbleStar_Q_N2"] = _two_body_to_ns_inv(
        params["k_Ar_dbleStar_Q_N2_m3_s"], loschmidt
    )
    params["K_Ar_4s_Q_CF4"] = _two_body_to_ns_inv(params["k_Ar_4s_Q_CF4_m3_s"], loschmidt)
    params["K_Ar_4s_Q_N2"] = _two_body_to_ns_inv(params["k_Ar_4s_Q_N2_m3_s"], loschmidt)
    params["K_Ar_4s_Q_2Ar"] = _three_body_to_ns_inv(params["k_Ar_4s_Q_2Ar_m6_s"], loschmidt)
    params["K_Ar2star_Q_CF4"] = _two_body_to_ns_inv(params["k_Ar2star_Q_CF4_m3_s"], loschmidt)
    params["K_Ar2star_Q_N2"] = _two_body_to_ns_inv(params["k_Ar2star_Q_N2_m3_s"], loschmidt)

    # Table 19 does not quote a separate bimolecular 4s+Ar coefficient. In the
    # compact TFM hypothesis, use the tabulated Ar**+Ar mean for the 4s cascade
    # step as well. This only fixes the competition against the additive.
    params["K_Ar_4s_Q_Ar"] = params["K_Ar_dbleStar_Q_Ar"]

    f1 = float(np.clip(float(params.get("f_1Sigma", 0.1)), 0.0, 1.0))
    f3 = float(np.clip(float(params.get("f_3Sigma", 1.0 - f1)), 0.0, 1.0))
    norm = f1 + f3
    if norm <= 0.0:
        f1, f3 = 0.1, 0.9
    else:
        f1, f3 = f1 / norm, f3 / norm
    params["f_1Sigma"] = f1
    params["f_3Sigma"] = f3
    # Backwards-compatible aliases used by old table helpers.
    params["f_S"] = f1
    params["f_S_Ar"] = f1
    params["f_S_CF4"] = f1
    params["f_S_ArCF4"] = f1
    params["f_S_N2"] = f1
    params["f_S_ArN2"] = f1

    fwhm_to_sigma = 1.0 / 2.354820045
    params["sigma_Ar2nd_nm"] = max(float(params["fwhm_Ar2nd_nm"]), 1.0e-12) * fwhm_to_sigma
    params["sigma_CF4_D_to_X_nm"] = max(float(params["fwhm_CF4_D_to_X_nm"]), 1.0e-12) * fwhm_to_sigma
    return params


def read_ar2nd_parameters(path: str | Path) -> dict[str, float]:
    """Read the name-based second-continuum parameter CSV."""
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
        value = row[value_col]
        if name and pd.notna(value):
            params[name] = float(value)
    return _finalise_parameters(params)


def _parameter_dict(params: Mapping[str, float] | pd.DataFrame | None) -> dict[str, float]:
    out = dict(DEFAULT_PARAMETERS)
    if params is None:
        return _finalise_parameters(out)
    if isinstance(params, pd.DataFrame):
        if "name" not in params.columns:
            raise ValueError("El DataFrame de parámetros debe contener una columna 'name'")
        numeric_cols = list(params.select_dtypes(include=["number"]).columns)
        value_col = "value" if "value" in params.columns else numeric_cols[0]
        for _, row in params.iterrows():
            if pd.notna(row[value_col]):
                out[str(row["name"]).strip()] = float(row[value_col])
    else:
        out.update({str(k): float(v) for k, v in params.items()})
    return _finalise_parameters(out)


def _prepare_fraction(f_additive):
    scalar_input = np.isscalar(f_additive) or np.asarray(f_additive).ndim == 0
    f = np.atleast_1d(np.asarray(f_additive, dtype=float))
    return np.clip(f, 0.0, 1.0), scalar_input


def _interp_values(degrad_data: pd.DataFrame, f_additive: np.ndarray, values: np.ndarray) -> np.ndarray:
    concentration = np.asarray(degrad_data["concentration"], dtype=float)
    idx = np.argsort(concentration)
    conc_sorted = concentration[idx]
    values_sorted = np.asarray(values, dtype=float)[idx]
    conc_unique, unique_idx = np.unique(conc_sorted, return_index=True)
    values_unique = values_sorted[unique_idx]
    if len(conc_unique) == 1:
        return np.repeat(values_unique[0], len(f_additive))

    # Outside the simulated Garfield concentration range, keep the nearest
    # simulated population fixed.  In particular, below the 0.1% CF4 point
    # this isolates the kinetic quenching extrapolation from an unsupported
    # extrapolation of the avalanche excitation populations.
    f_eval = np.clip(np.asarray(f_additive, dtype=float), conc_unique[0], conc_unique[-1])
    return np.asarray(
        PchipInterpolator(conc_unique, values_unique, extrapolate=False)(f_eval),
        dtype=float,
    )


def _interp_column(degrad_data: pd.DataFrame, f_additive: np.ndarray, candidates: tuple[str, ...]) -> np.ndarray:
    """Interpolate the first available alias, never sum aliases.

    The candidate tuples contain alternative names for the same physical
    population.  In particular, ``Ar_2nd_precursor`` is a legacy aggregate
    and can coexist with the three resolved bins.  Summing all candidates
    would therefore count the same excitations twice.
    """
    existing = next((col for col in candidates if col in degrad_data.columns), None)
    if existing is None:
        return np.zeros_like(f_additive, dtype=float)
    return _interp_values(
        degrad_data,
        f_additive,
        np.asarray(degrad_data[existing], dtype=float),
    )


def _interp_sum_columns(degrad_data: pd.DataFrame, f_additive: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    existing = [col for col in columns if col in degrad_data.columns]
    if not existing:
        return np.zeros_like(f_additive, dtype=float)
    values = np.zeros(len(degrad_data), dtype=float)
    for col in existing:
        values += np.asarray(degrad_data[col], dtype=float)
    return _interp_values(degrad_data, f_additive, values)


def _positive_rate(value: float) -> float:
    value = float(value)
    return max(value, 0.0) if np.isfinite(value) else 0.0


def _population_components(
    params: Mapping[str, float],
    degrad_data: pd.DataFrame,
    f_additive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return N_Ar**, N_Ar(1s2,1s3), N_Ar(1s4,1s5)."""
    n_upper = np.clip(_interp_column(degrad_data, f_additive, AR_UPPER_CANDIDATES), 0.0, None)
    n_upper *= float(params.get("W_Ar_dbleStar_to_1s", 1.0))

    n_high_4s = np.clip(_interp_column(degrad_data, f_additive, AR_HIGH_4S_CANDIDATES), 0.0, None)
    n_precursor = np.clip(
        _interp_column(degrad_data, f_additive, AR_EXCIMER_PRECURSOR_CANDIDATES), 0.0, None
    )

    # Backwards compatibility with the previous dedicated tables: Ar_meta and
    # Ar_res were exactly the 1s5 and 1s4 bins, respectively.
    if not any(col in degrad_data.columns for col in AR_EXCIMER_PRECURSOR_CANDIDATES):
        n_precursor = np.clip(
            _interp_sum_columns(degrad_data, f_additive, ("Ar_meta", "Ar_res")),
            0.0,
            None,
        )
    return n_upper, n_high_4s, n_precursor


def _effective_ar_4s_population(
    params: Mapping[str, float],
    degrad_data: pd.DataFrame,
    f_additive: np.ndarray,
) -> np.ndarray:
    """Raw population entering the compact second-continuum cascade."""
    n_upper, n_high_4s, n_precursor = _population_components(params, degrad_data, f_additive)
    return n_upper + n_high_4s + n_precursor


def _effective_ar_4s_components(
    params: Mapping[str, float],
    degrad_data: pd.DataFrame,
    f_additive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper returning precursor, upper-4s and Ar** populations."""
    n_upper, n_high_4s, n_precursor = _population_components(params, degrad_data, f_additive)
    return n_precursor, n_high_4s, n_upper


def _singlet_fraction(params: Mapping[str, float], gas_mixture: str) -> float:
    del gas_mixture
    return float(np.clip(float(params.get("f_1Sigma", 0.1)), 0.0, 1.0))


def _singlet_fraction_for_additive(
    params: Mapping[str, float], gas_mixture: str, f_additive: np.ndarray
) -> np.ndarray:
    return np.full_like(np.asarray(f_additive, dtype=float), _singlet_fraction(params, gas_mixture))


def _additive_rates(params: Mapping[str, float], gas_mixture: str) -> tuple[float, float, float]:
    """Return Ar**, Ar(4s), and Ar2* additive rates [ns-1 at n=1]."""
    gas_key = gas_mixture.upper().replace("-", "")
    if "CF4" in gas_key:
        return (
            _positive_rate(params["K_Ar_dbleStar_Q_CF4"]),
            _positive_rate(params["K_Ar_4s_Q_CF4"]),
            _positive_rate(params["K_Ar2star_Q_CF4"]),
        )
    if "N2" in gas_key:
        return (
            _positive_rate(params["K_Ar_dbleStar_Q_N2"]),
            _positive_rate(params["K_Ar_4s_Q_N2"]),
            _positive_rate(params["K_Ar2star_Q_N2"]),
        )
    return 0.0, 0.0, 0.0


def _upper_cascade_probability(
    params: Mapping[str, float], f_additive: np.ndarray, n: float, k_add: float
) -> np.ndarray:
    f_ar = 1.0 - f_additive
    gamma = 1.0 / max(float(params["tau_Ar_dbleStar_ns"]), 1.0e-30)
    k_ar = _positive_rate(params["K_Ar_dbleStar_Q_Ar"])
    productive = gamma + n * f_ar * k_ar
    denominator = productive + n * f_additive * k_add
    return np.divide(productive, denominator, out=np.zeros_like(productive), where=denominator > 0.0)


def _high_4s_transfer_probability(
    params: Mapping[str, float], f_additive: np.ndarray, n: float, k_add: float
) -> np.ndarray:
    f_ar = 1.0 - f_additive
    productive = n * f_ar * _positive_rate(params["K_Ar_4s_Q_Ar"])
    denominator = productive + n * f_additive * k_add
    return np.divide(productive, denominator, out=np.zeros_like(productive), where=denominator > 0.0)


def _formation_probability(
    params: Mapping[str, float], f_additive: np.ndarray, n: float, k_add: float
) -> np.ndarray:
    f_ar = 1.0 - f_additive
    productive = (n**2) * (f_ar**2) * _positive_rate(params["K_Ar_4s_Q_2Ar"])
    denominator = productive + n * f_additive * k_add
    return np.divide(productive, denominator, out=np.zeros_like(productive), where=denominator > 0.0)


def _radiative_survival(
    params: Mapping[str, float], f_additive: np.ndarray, n: float, k_add: float
) -> tuple[np.ndarray, np.ndarray]:
    inv_tau_s = 1.0 / max(float(params["tau_S_ns"]), 1.0e-30)
    inv_tau_t = 1.0 / max(float(params["tau_T_ns"]), 1.0e-30)
    collisional = n * f_additive * k_add
    den_s = inv_tau_s + collisional
    den_t = inv_tau_t + collisional
    p_s = np.divide(inv_tau_s, den_s, out=np.zeros_like(f_additive), where=den_s > 0.0)
    p_t = np.divide(inv_tau_t, den_t, out=np.zeros_like(f_additive), where=den_t > 0.0)
    return p_s, p_t


def _base_yield_per_keV(
    params: Mapping[str, float],
    degrad_data: pd.DataFrame,
    f_additive: np.ndarray,
    n: float,
    *,
    gas_mixture: str,
    energy_xray_kev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast and slow second-continuum yields per keV from the TFM equation."""
    k_upper_add, k_4s_add, k_excimer_add = _additive_rates(params, gas_mixture)
    # Use the actual additive fraction continuously.  Pure argon is obtained
    # only at f_additive = 0; no numerical switch is applied near zero, which
    # avoids an artificial discontinuity in the 99.9--99.999% Ar extrapolation.
    f_kinetic = np.asarray(f_additive, dtype=float)

    n_upper, n_high_4s, n_precursor = _population_components(params, degrad_data, f_additive)
    p_upper = _upper_cascade_probability(params, f_kinetic, n, k_upper_add)
    p_high_4s = _high_4s_transfer_probability(params, f_kinetic, n, k_4s_add)
    p_form = _formation_probability(params, f_kinetic, n, k_4s_add)
    p_rad_s, p_rad_t = _radiative_survival(params, f_kinetic, n, k_excimer_add)

    n_fed = n_upper * p_upper + n_high_4s * p_high_4s + n_precursor
    n_excimer = n_fed * p_form
    energy = max(float(energy_xray_kev), 1.0e-30)
    singlet = n_excimer * float(params["f_1Sigma"]) * p_rad_s / energy
    triplet = n_excimer * float(params["f_3Sigma"]) * p_rad_t / energy
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
    """Ar second-continuum yield using the compact model in Sec. 10.1.3.

    ``n`` is p/(1 bar). ``energy_xray_ev`` keeps the historical argument name,
    but its value is in keV in this project. Nnorm is intentionally ignored.
    """
    del n_norm
    p = _parameter_dict(params)
    f, scalar_input = _prepare_fraction(f_additive)
    n = max(float(n), 0.0)

    singlet_base, triplet_base = _base_yield_per_keV(
        p,
        degrad_data,
        f,
        n,
        gas_mixture=gas_mixture,
        energy_xray_kev=float(energy_xray_ev),
    )
    scale = float(p.get("scale_Ar2nd", 1.0))
    singlet = singlet_base * scale
    triplet = triplet_base * scale
    triplet_weight = float(np.clip(float(p.get("triplet_weight", 0.0)), 0.0, 1.0))
    total = singlet + triplet_weight * triplet

    # Legacy optional anchor retained only for explicit sensitivity studies.
    if float(p.get("anchor_Ar2nd_to_pure_argon", 0.0)) > 0.5:
        ref_f = np.asarray([float(p.get("reference_additive_fraction", 1.0e-5))])
        ref_s, ref_t = _base_yield_per_keV(
            p,
            degrad_data,
            ref_f,
            float(p.get("reference_pressure_bar", 1.0)),
            gas_mixture=gas_mixture,
            energy_xray_kev=float(energy_xray_ev),
        )
        ref = float((ref_s + triplet_weight * ref_t)[0] * scale)
        component_fraction = float(p["f_1Sigma"] + triplet_weight * p["f_3Sigma"])
        target_raw = float(p.get("Y_Ar2nd_pure_ph_MeV", 1.47e4)) * component_fraction / 1.0e3
        if np.isfinite(ref) and ref > 0.0:
            factor = target_raw / ref
            singlet *= factor
            triplet *= factor
            total *= factor

    if activate_components:
        if scalar_input:
            return total.item(), singlet.item(), triplet.item()
        return total, singlet, triplet
    return total.item() if scalar_input else total
