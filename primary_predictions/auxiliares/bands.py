from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_band(samples: np.ndarray, central: np.ndarray, percentiles=(16.0, 84.0)) -> dict[str, np.ndarray]:
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
        }

    low, high = np.nanpercentile(samples, percentiles, axis=0)
    minus = np.clip(central - low, 0.0, None)
    plus = np.clip(high - central, 0.0, None)
    sigma = 0.5 * (minus + plus)
    return {
        "low": low,
        "high": high,
        "minus": minus,
        "plus": plus,
        "sigma": sigma,
    }


def combine_stat_syst(central: np.ndarray, stat: dict[str, np.ndarray], syst: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    central = np.asarray(central, dtype=float)
    stat_minus = np.nan_to_num(stat["minus"], nan=0.0)
    stat_plus = np.nan_to_num(stat["plus"], nan=0.0)
    syst_minus = np.nan_to_num(syst["minus"], nan=0.0)
    syst_plus = np.nan_to_num(syst["plus"], nan=0.0)

    minus = np.sqrt(stat_minus**2 + syst_minus**2)
    plus = np.sqrt(stat_plus**2 + syst_plus**2)
    return {
        "low": central - minus,
        "high": central + plus,
        "minus": minus,
        "plus": plus,
        "sigma": 0.5 * (minus + plus),
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
        }
    )


def asymmetric_errors(central: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    band = percentile_band(samples, central)
    return band["minus"], band["plus"]

