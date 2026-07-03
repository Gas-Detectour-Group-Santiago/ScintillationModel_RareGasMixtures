from __future__ import annotations

import numpy as np
import pandas as pd


def _nanpercentile_1d_safe(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.nanpercentile(values, q))


def percentile_band(samples: np.ndarray, central: np.ndarray, percentiles=(16.0, 84.0)) -> dict[str, np.ndarray]:
    """Return a toy uncertainty band centred on the optimal prediction.

    Primary-prediction plots are shown on a logarithmic y-axis and the toy
    variations behave mostly as multiplicative uncertainties.  A raw percentile
    interval, or even an asymmetric absolute interval around the best fit, can
    make the optimal curve sit close to one edge of the shaded band.  That is
    visually misleading for the primary-fit figures, where the shaded region is
    meant to represent ``stat/sys around the optimal curve``.

    The displayed band is therefore built in log-ratio space around the optimal
    prediction:

        log(samples / central) -> 16/84 percentile spread.

    The larger side of that spread is mirrored so that the optimal prediction is
    the geometric centre of the band.  This keeps the band multiplicative,
    positive, and visually centred on log plots.  Raw linear percentiles are
    still exported for diagnostics through ``raw_low``/``raw_high`` and
    ``central_position``.
    """

    central = np.asarray(central, dtype=float)
    samples = np.asarray(samples, dtype=float)

    nan = np.full_like(central, np.nan, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return {
            "low": nan,
            "high": nan,
            "minus": nan,
            "plus": nan,
            "sigma": nan,
            "raw_low": nan,
            "raw_high": nan,
            "central_position": nan,
            "log_minus": nan,
            "log_plus": nan,
            "log_sigma": nan,
        }

    raw_low, raw_high = np.nanpercentile(samples, percentiles, axis=0)

    log_sigma = np.full_like(central, np.nan, dtype=float)
    qlo_arr = np.full_like(central, np.nan, dtype=float)
    qhi_arr = np.full_like(central, np.nan, dtype=float)

    for j, c in enumerate(central):
        sj = samples[:, j]
        valid = np.isfinite(sj) & (sj > 0.0) & np.isfinite(c) & (c > 0.0)
        if np.any(valid):
            log_ratio = np.log(sj[valid] / c)
            qlo = _nanpercentile_1d_safe(log_ratio, float(percentiles[0]))
            qhi = _nanpercentile_1d_safe(log_ratio, float(percentiles[1]))
            qlo_arr[j] = qlo
            qhi_arr[j] = qhi
            log_sigma[j] = max(abs(qlo), abs(qhi))

    # Fallback for non-positive/invalid points: use a symmetric absolute band.
    linear_minus = np.abs(central - raw_low)
    linear_plus = np.abs(raw_high - central)
    linear_sigma = np.maximum(linear_minus, linear_plus)

    low = np.empty_like(central, dtype=float)
    high = np.empty_like(central, dtype=float)
    log_ok = np.isfinite(log_sigma) & np.isfinite(central) & (central > 0.0)
    low[log_ok] = central[log_ok] * np.exp(-log_sigma[log_ok])
    high[log_ok] = central[log_ok] * np.exp(log_sigma[log_ok])
    low[~log_ok] = central[~log_ok] - linear_sigma[~log_ok]
    high[~log_ok] = central[~log_ok] + linear_sigma[~log_ok]

    minus = central - low
    plus = high - central
    sigma = 0.5 * (minus + plus)

    denom = raw_high - raw_low
    with np.errstate(divide="ignore", invalid="ignore"):
        central_position = (central - raw_low) / denom
    central_position = np.where(np.isfinite(central_position), central_position, np.nan)

    return {
        "low": low,
        "high": high,
        "minus": minus,
        "plus": plus,
        "sigma": sigma,
        "raw_low": raw_low,
        "raw_high": raw_high,
        "central_position": central_position,
        "log_minus": log_sigma,
        "log_plus": log_sigma,
        "log_sigma": log_sigma,
        "raw_log_p16": qlo_arr,
        "raw_log_p84": qhi_arr,
    }


def combine_stat_syst(central: np.ndarray, stat: dict[str, np.ndarray], syst: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    central = np.asarray(central, dtype=float)

    stat_log = np.asarray(stat.get("log_sigma", np.full_like(central, np.nan)), dtype=float)
    syst_log = np.asarray(syst.get("log_sigma", np.full_like(central, np.nan)), dtype=float)
    log_ok = (
        np.isfinite(central)
        & (central > 0.0)
        & (np.isfinite(stat_log) | np.isfinite(syst_log))
    )

    minus = np.full_like(central, np.nan, dtype=float)
    plus = np.full_like(central, np.nan, dtype=float)
    low = np.full_like(central, np.nan, dtype=float)
    high = np.full_like(central, np.nan, dtype=float)

    if np.any(log_ok):
        total_log = np.sqrt(np.nan_to_num(stat_log[log_ok], nan=0.0) ** 2 + np.nan_to_num(syst_log[log_ok], nan=0.0) ** 2)
        low[log_ok] = central[log_ok] * np.exp(-total_log)
        high[log_ok] = central[log_ok] * np.exp(total_log)
        minus[log_ok] = central[log_ok] - low[log_ok]
        plus[log_ok] = high[log_ok] - central[log_ok]

    if np.any(~log_ok):
        stat_minus = np.nan_to_num(stat["minus"], nan=0.0)
        stat_plus = np.nan_to_num(stat["plus"], nan=0.0)
        syst_minus = np.nan_to_num(syst["minus"], nan=0.0)
        syst_plus = np.nan_to_num(syst["plus"], nan=0.0)
        minus[~log_ok] = np.sqrt(stat_minus[~log_ok] ** 2 + syst_minus[~log_ok] ** 2)
        plus[~log_ok] = np.sqrt(stat_plus[~log_ok] ** 2 + syst_plus[~log_ok] ** 2)
        low[~log_ok] = central[~log_ok] - minus[~log_ok]
        high[~log_ok] = central[~log_ok] + plus[~log_ok]

    return {
        "low": low,
        "high": high,
        "minus": minus,
        "plus": plus,
        "sigma": 0.5 * (minus + plus),
        "log_sigma": np.where(log_ok, np.sqrt(np.nan_to_num(stat_log, nan=0.0) ** 2 + np.nan_to_num(syst_log, nan=0.0) ** 2), np.nan),
    }


def band_dataframe(x, central, stat, syst, total, *, x_name="concentration") -> pd.DataFrame:
    return pd.DataFrame(
        {
            x_name: np.asarray(x, dtype=float),
            "central": np.asarray(central, dtype=float),
            "stat_low": stat["low"],
            "stat_high": stat["high"],
            "syst_low": syst["low"],
            "syst_high": syst["high"],
            "total_low": total["low"],
            "total_high": total["high"],
            "stat_raw_p16": stat.get("raw_low"),
            "stat_raw_p84": stat.get("raw_high"),
            "stat_central_position": stat.get("central_position"),
            "syst_raw_p16": syst.get("raw_low"),
            "syst_raw_p84": syst.get("raw_high"),
            "syst_central_position": syst.get("central_position"),
            "stat_log_sigma": stat.get("log_sigma"),
            "syst_log_sigma": syst.get("log_sigma"),
            "total_log_sigma": total.get("log_sigma"),
            "stat_raw_log_p16": stat.get("raw_log_p16"),
            "stat_raw_log_p84": stat.get("raw_log_p84"),
            "syst_raw_log_p16": syst.get("raw_log_p16"),
            "syst_raw_log_p84": syst.get("raw_log_p84"),
        }
    )


def asymmetric_errors(central: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    band = percentile_band(samples, central)
    return band["minus"], band["plus"]
