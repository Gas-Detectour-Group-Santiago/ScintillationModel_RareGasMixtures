from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# Loschmidt number used in the TFM (ideal gas at the reference state).
LOSCHMIDT_M3 = 2.6868e25
SECONDS_PER_NANOSECOND = 1.0e-9


ADDITIVE_PARAMETER_NAMES: tuple[str, ...] = (
    "k_Ar_dbleStar_Q_m3_s",
    "k_Ar_4s_Q_m3_s",
    "k_Ar2star_Q_m3_s",
)


def normalise_additive_name(name: str | None) -> str:
    """Return the canonical additive key used by the parameter CSV."""
    if name is None:
        return ""
    compact = str(name).strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "ARGON": "AR",
        "NITROGEN": "N2",
        "CARBONDIOXIDE": "CO2",
        "METHANE": "CH4",
    }
    return aliases.get(compact, compact)


def additive_from_mixture(gas_mixture: str | None) -> str:
    """Infer the additive from names such as ArCF4, Ar-N2, ArCO2 or ArCH4."""
    key = normalise_additive_name(gas_mixture)
    if key in {"", "AR", "PUREAR", "PUREARGON"}:
        return ""
    if key.startswith("AR") and len(key) > 2:
        return normalise_additive_name(key[2:])
    return key


def _additive_storage_key(parameter_name: str, additive: str) -> str:
    return f"{parameter_name}__{normalise_additive_name(additive)}"


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
    "k_Ar_4s_Q_2Ar_m6_s": 1.00e-44,
    # Built-in fallbacks keep old scripts working when no CSV is supplied.
    "k_Ar_dbleStar_Q_m3_s__CF4": 1.80e-16,
    "k_Ar_4s_Q_m3_s__CF4": 3.00e-17,
    "k_Ar2star_Q_m3_s__CF4": 3.00e-17,
    "k_Ar_dbleStar_Q_m3_s__N2": 1.58e-16,
    "k_Ar_4s_Q_m3_s__N2": 3.60e-17,
    "k_Ar2star_Q_m3_s__N2": 2.24e-18,
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


def _finalise_parameters(params: dict[str, object]) -> dict[str, object]:
    """Derive pressure-normalised rates and expose every CSV additive dynamically."""
    loschmidt = max(float(params.get("loschmidt_m3", LOSCHMIDT_M3)), 0.0)

    params["K_Ar_dbleStar_Q_Ar"] = _two_body_to_ns_inv(
        params["k_Ar_dbleStar_Q_Ar_m3_s"], loschmidt
    )
    params["K_Ar_4s_Q_Ar"] = _two_body_to_ns_inv(
        params.get("k_Ar_4s_Q_Ar_m3_s", params["k_Ar_dbleStar_Q_Ar_m3_s"]),
        loschmidt,
    )
    params["K_Ar_4s_Q_2Ar"] = _three_body_to_ns_inv(
        params["k_Ar_4s_Q_2Ar_m6_s"], loschmidt
    )

    additives: set[str] = set()
    for key in tuple(params):
        if "__" not in key:
            continue
        base, additive = key.rsplit("__", 1)
        additive = normalise_additive_name(additive)
        if base in ADDITIVE_PARAMETER_NAMES and additive:
            additives.add(additive)

    for additive in sorted(additives):
        for source_name, target_name in (
            ("k_Ar_dbleStar_Q_m3_s", "K_Ar_dbleStar_Q"),
            ("k_Ar_4s_Q_m3_s", "K_Ar_4s_Q"),
            ("k_Ar2star_Q_m3_s", "K_Ar2star_Q"),
        ):
            source = _additive_storage_key(source_name, additive)
            if source not in params:
                raise ValueError(
                    f"Falta {source_name!r} para el aditivo {additive!r} en el CSV del segundo continuo"
                )
            params[_additive_storage_key(target_name, additive)] = _two_body_to_ns_inv(
                params[source], loschmidt
            )

        # Compatibility aliases for the two historical gases.
        params[f"K_Ar_dbleStar_Q_{additive}"] = params[
            _additive_storage_key("K_Ar_dbleStar_Q", additive)
        ]
        params[f"K_Ar_4s_Q_{additive}"] = params[
            _additive_storage_key("K_Ar_4s_Q", additive)
        ]
        params[f"K_Ar2star_Q_{additive}"] = params[
            _additive_storage_key("K_Ar2star_Q", additive)
        ]

    params["available_additives"] = tuple(sorted(additives))

    f1 = float(np.clip(float(params.get("f_1Sigma", 0.1)), 0.0, 1.0))
    f3 = float(np.clip(float(params.get("f_3Sigma", 1.0 - f1)), 0.0, 1.0))
    norm = f1 + f3
    if norm <= 0.0:
        f1, f3 = 0.1, 0.9
    else:
        f1, f3 = f1 / norm, f3 / norm
    params["f_1Sigma"] = f1
    params["f_3Sigma"] = f3
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


def available_additives(params: Mapping[str, object]) -> tuple[str, ...]:
    values = params.get("available_additives", ())
    return tuple(str(value) for value in values)


def _load_parameter_frame(df: pd.DataFrame, base: dict[str, object]) -> dict[str, object]:
    """Load both the new long CSV and the legacy name/value layout."""
    if "name" not in df.columns:
        raise ValueError("El CSV de parámetros debe contener una columna 'name'")
    value_col = "value" if "value" in df.columns else None
    if value_col is None:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            raise ValueError("El CSV de parámetros debe contener una columna numérica de valores")
        value_col = str(numeric.columns[0])

    is_long = {"scope", "additive"}.issubset(df.columns)
    for _, row in df.iterrows():
        enabled = str(row.get("enabled", "true")).strip().lower() not in {"0", "false", "no", "off"}
        if not enabled:
            continue
        name = str(row["name"]).strip()
        value = row[value_col]
        if not name or pd.isna(value):
            continue
        if is_long and str(row.get("scope", "common")).strip().lower() == "additive":
            additive = normalise_additive_name(str(row.get("additive", "")))
            if not additive:
                raise ValueError(f"Fila aditiva sin gas para el parámetro {name!r}")
            base[_additive_storage_key(name, additive)] = float(value)
            continue

        # Legacy additive names are translated into the new generic storage.
        legacy = re.match(
            r"^(k_Ar_dbleStar_Q|k_Ar_4s_Q|k_Ar2star_Q)_([A-Za-z0-9]+)_m3_s$",
            name,
        )
        if legacy and legacy.group(2).upper() != "AR":
            generic = f"{legacy.group(1)}_m3_s"
            base[_additive_storage_key(generic, legacy.group(2))] = float(value)
        else:
            base[name] = float(value)
    return base


def read_ar2nd_parameters(path: str | Path) -> dict[str, object]:
    """Read the common and additive-specific second-continuum parameter database."""
    params: dict[str, object] = dict(DEFAULT_PARAMETERS)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el CSV de parámetros del segundo continuo: {path}")
    params = _load_parameter_frame(pd.read_csv(path), params)
    return _finalise_parameters(params)


def _parameter_dict(params: Mapping[str, object] | pd.DataFrame | None) -> dict[str, object]:
    out: dict[str, object] = dict(DEFAULT_PARAMETERS)
    if params is None:
        return _finalise_parameters(out)
    if isinstance(params, pd.DataFrame):
        out = _load_parameter_frame(params, out)
    else:
        for key, value in params.items():
            if key == "available_additives":
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                out[str(key)] = float(value)
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




def validate_additive_parameters(params: Mapping[str, object], additive: str) -> None:
    """Raise a clear error unless all required additive rates are available."""
    additive_key = normalise_additive_name(additive)
    if not additive_key:
        return
    required = tuple(_additive_storage_key(name, additive_key) for name in ADDITIVE_PARAMETER_NAMES)
    missing = [key for key in required if key not in params or not np.isfinite(float(params[key]))]
    if missing:
        known = ", ".join(available_additives(params)) or "none"
        raise ValueError(
            f"Incomplete Ar second-continuum kinetics for {additive_key}. "
            f"Available additives: {known}. Missing: {', '.join(missing)}"
        )


def _additive_rates(
    params: Mapping[str, object],
    gas_mixture: str | None,
    additive: str | None = None,
) -> tuple[float, float, float]:
    """Return Ar**, Ar(4s), and Ar2* rates selected from the CSV database."""
    additive_key = normalise_additive_name(additive) or additive_from_mixture(gas_mixture)
    if not additive_key:
        return 0.0, 0.0, 0.0

    validate_additive_parameters(params, additive_key)
    names = ("K_Ar_dbleStar_Q", "K_Ar_4s_Q", "K_Ar2star_Q")
    keys = tuple(_additive_storage_key(name, additive_key) for name in names)
    return tuple(_positive_rate(float(params[key])) for key in keys)


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
    gas_mixture: str | None,
    additive: str | None = None,
    energy_xray_kev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast and slow second-continuum yields per keV from the TFM equation."""
    k_upper_add, k_4s_add, k_excimer_add = _additive_rates(params, gas_mixture, additive)
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
    gas_mixture: str | None = None,
    additive: str | None = None,
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
        additive=additive,
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
            additive=additive,
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
