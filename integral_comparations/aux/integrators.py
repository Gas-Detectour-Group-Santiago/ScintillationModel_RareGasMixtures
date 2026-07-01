from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

IntegralMethod = Literal["trapz", "simpson", "quad", "gaussian_fit"]
GaussianCenterMode = Literal["auto", "fixed", "bounded", "free", "shared_shift"]
GaussianWidthMode = Literal["shared", "individual"]
GaussianBaselineMode = Literal["none", "constant", "linear"]


@dataclass(frozen=True)
class IntegralConfig:
    """Configuration for one spectral integral.

    Standard methods:
        - ``trapz``: hard cut + trapezoidal integration.
        - ``simpson``: hard cut + Simpson integration.
        - ``quad``: hard cut + scipy.quad over a linear interpolation.

    Gaussian-fit method:
        - ``gaussian_fit``: fit one or several Gaussians in the hard-cut window.

    Gaussian options:
        - ``gaussian_centers_nm``: nominal peak centres.
        - ``gaussian_peak_names``: optional labels for diagnostics/plots.
        - ``gaussian_center_mode``:
            * ``auto``: one Gaussian centred at the local maximum.
            * ``fixed``: centres fixed to ``gaussian_centers_nm``.
            * ``bounded``: each centre moves independently within
              ``±gaussian_center_tolerance_nm``.
            * ``free``: each centre is free inside the integration window.
            * ``shared_shift``: all centres move together with one common shift
              bounded by ``±gaussian_center_tolerance_nm``.
        - ``gaussian_width_mode``:
            * ``shared``: all peaks share one sigma.
            * ``individual``: each peak has its own sigma.
        - ``gaussian_sigma_nm``: initial sigma guess. Scalar or one per peak.
        - ``gaussian_sigma_bounds_nm``: hard bounds for sigma during the fit.
        - ``gaussian_baseline_mode``: none / constant / linear baseline.
        - ``gaussian_positive_amplitudes``: constrain amplitudes >= 0.
        - ``gaussian_integrate_baseline``: include fitted baseline in the final
          integral. Normally False.
    """

    wavelength_range_nm: tuple[float, float]
    method: IntegralMethod = "trapz"
    gaussian_centers_nm: tuple[float, ...] = ()
    gaussian_peak_names: tuple[str, ...] = ()
    gaussian_center_mode: GaussianCenterMode = "auto"
    gaussian_center_tolerance_nm: float = 1.0
    gaussian_width_mode: GaussianWidthMode = "shared"
    gaussian_sigma_nm: float | tuple[float, ...] = 8.0
    gaussian_sigma_bounds_nm: tuple[float, float] = (0.1, 40.0)
    gaussian_baseline_mode: GaussianBaselineMode = "constant"
    gaussian_positive_amplitudes: bool = True
    gaussian_integrate_baseline: bool = False
    clip_negative: bool = True


@dataclass(frozen=True)
class IntegralResult:
    value: float
    method: str
    xmin_nm: float
    xmax_nm: float
    n_points: int
    metadata: dict[str, object] = field(default_factory=dict)
    plot_payload: dict[str, object] | None = None


def _clean_sorted_arrays(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    *,
    clip_negative: bool,
) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(wavelength_nm, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    n = min(w.size, y.size)
    w = w[:n]
    y = y[:n]

    finite = np.isfinite(w) & np.isfinite(y)
    w = w[finite]
    y = y[finite]
    if clip_negative:
        y = np.clip(y, 0.0, None)

    if w.size < 2:
        return w, y

    order = np.argsort(w)
    w = w[order]
    y = y[order]

    w_unique, unique_idx = np.unique(w, return_index=True)
    y_unique = y[unique_idx]
    return w_unique, y_unique


def _hardcut_with_boundaries(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    xmin: float,
    xmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    if xmax <= xmin or wavelength_nm.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    if xmax <= wavelength_nm[0] or xmin >= wavelength_nm[-1]:
        return np.array([], dtype=float), np.array([], dtype=float)

    xmin_eff = max(float(xmin), float(wavelength_nm[0]))
    xmax_eff = min(float(xmax), float(wavelength_nm[-1]))

    mask = (wavelength_nm >= xmin_eff) & (wavelength_nm <= xmax_eff)
    w_int = wavelength_nm[mask]
    y_int = intensity[mask]

    y_xmin = np.interp(xmin_eff, wavelength_nm, intensity)
    y_xmax = np.interp(xmax_eff, wavelength_nm, intensity)

    w_int = np.concatenate(([xmin_eff], w_int, [xmax_eff]))
    y_int = np.concatenate(([y_xmin], y_int, [y_xmax]))

    order = np.argsort(w_int)
    w_int = w_int[order]
    y_int = y_int[order]

    w_int, unique_idx = np.unique(w_int, return_index=True)
    y_int = y_int[unique_idx]
    return w_int, y_int


def integrate_spectrum(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    config: IntegralConfig,
) -> IntegralResult:
    xmin, xmax = map(float, config.wavelength_range_nm)
    w, y = _clean_sorted_arrays(wavelength_nm, intensity, clip_negative=config.clip_negative)
    if w.size < 2:
        return IntegralResult(np.nan, config.method, xmin, xmax, 0, {"reason": "not_enough_points"})

    if config.method == "trapz":
        return _integrate_trapz(w, y, xmin, xmax, config)
    if config.method == "simpson":
        return _integrate_simpson(w, y, xmin, xmax, config)
    if config.method == "quad":
        return _integrate_quad(w, y, xmin, xmax, config)
    if config.method == "gaussian_fit":
        return _integrate_gaussian_fit(w, y, xmin, xmax, config)

    raise ValueError(f"Unknown integral method: {config.method!r}")


def _integrate_trapz(w: np.ndarray, y: np.ndarray, xmin: float, xmax: float, config: IntegralConfig) -> IntegralResult:
    w_int, y_int = _hardcut_with_boundaries(w, y, xmin, xmax)
    if w_int.size < 2:
        return IntegralResult(np.nan, "trapz", xmin, xmax, int(w_int.size), {"reason": "empty_window"})
    return IntegralResult(
        float(np.trapezoid(y_int, w_int)),
        "trapz",
        xmin,
        xmax,
        int(w_int.size),
        {},
    )


def _integrate_simpson(w: np.ndarray, y: np.ndarray, xmin: float, xmax: float, config: IntegralConfig) -> IntegralResult:
    w_int, y_int = _hardcut_with_boundaries(w, y, xmin, xmax)
    if w_int.size < 2:
        return IntegralResult(np.nan, "simpson", xmin, xmax, int(w_int.size), {"reason": "empty_window"})
    try:
        from scipy.integrate import simpson

        value = float(simpson(y_int, x=w_int))
        method = "simpson"
        metadata: dict[str, object] = {}
    except Exception as exc:
        value = float(np.trapezoid(y_int, w_int))
        method = "simpson_fallback_trapz"
        metadata = {"fallback_reason": type(exc).__name__}
    return IntegralResult(value, method, xmin, xmax, int(w_int.size), metadata)



def _integrate_quad(w: np.ndarray, y: np.ndarray, xmin: float, xmax: float, config: IntegralConfig) -> IntegralResult:
    if xmax <= w[0] or xmin >= w[-1]:
        return IntegralResult(np.nan, "quad", xmin, xmax, 0, {"reason": "empty_window"})
    xmin_eff = max(xmin, float(w[0]))
    xmax_eff = min(xmax, float(w[-1]))
    try:
        from scipy.integrate import quad

        def func(xx: float) -> float:
            return float(np.interp(xx, w, y))

        value, error = quad(func, xmin_eff, xmax_eff, limit=200)
        return IntegralResult(
            float(value),
            "quad_linear_interp",
            xmin,
            xmax,
            int(((w >= xmin_eff) & (w <= xmax_eff)).sum()),
            {"quad_error": float(error)},
        )
    except Exception as exc:
        fallback = _integrate_trapz(w, y, xmin, xmax, config)
        metadata = dict(fallback.metadata)
        metadata["fallback_reason"] = type(exc).__name__
        return IntegralResult(fallback.value, "quad_fallback_trapz", xmin, xmax, fallback.n_points, metadata)


# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------

def _normal_cdf(z: np.ndarray | float) -> np.ndarray | float:
    from math import erf, sqrt

    z_arr = np.asarray(z, dtype=float)
    out = 0.5 * (1.0 + np.vectorize(erf)(z_arr / sqrt(2.0)))
    return float(out) if np.isscalar(z) else out


def _gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1.0e-12)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


@dataclass(frozen=True)
class _GaussianModelSpec:
    baseline_mode: GaussianBaselineMode
    center_mode: GaussianCenterMode
    width_mode: GaussianWidthMode
    peak_names: tuple[str, ...]
    nominal_centers: tuple[float, ...]
    center_tolerance_nm: float
    sigma_init: tuple[float, ...]
    sigma_bounds_nm: tuple[float, float]
    positive_amplitudes: bool


@dataclass(frozen=True)
class _FitEval:
    total_y: np.ndarray
    baseline_y: np.ndarray
    component_ys: list[np.ndarray]
    centres_nm: list[float]
    sigmas_nm: list[float]
    amplitudes: list[float]



def _resolve_centers_and_sigmas(config: IntegralConfig, w_int: np.ndarray, y_int: np.ndarray) -> tuple[tuple[float, ...], tuple[str, ...], tuple[float, ...]]:
    centers = tuple(float(c) for c in config.gaussian_centers_nm)
    if config.gaussian_center_mode == "auto" or not centers:
        centers = (float(w_int[np.nanargmax(y_int)]),)
    peak_names = tuple(config.gaussian_peak_names) if config.gaussian_peak_names else tuple(f"peak_{i}" for i in range(len(centers)))
    if len(peak_names) != len(centers):
        peak_names = tuple(f"peak_{i}" for i in range(len(centers)))

    if isinstance(config.gaussian_sigma_nm, Sequence) and not isinstance(config.gaussian_sigma_nm, (str, bytes)):
        sigmas = tuple(float(s) for s in config.gaussian_sigma_nm)
        if len(sigmas) != len(centers):
            raise ValueError("gaussian_sigma_nm sequence must match gaussian_centers_nm length")
    else:
        sigmas = tuple(float(config.gaussian_sigma_nm) for _ in centers)
    return centers, peak_names, sigmas



def _build_gaussian_model_spec(config: IntegralConfig, w_int: np.ndarray, y_int: np.ndarray) -> _GaussianModelSpec:
    centers, peak_names, sigmas = _resolve_centers_and_sigmas(config, w_int, y_int)
    return _GaussianModelSpec(
        baseline_mode=config.gaussian_baseline_mode,
        center_mode=config.gaussian_center_mode,
        width_mode=config.gaussian_width_mode,
        peak_names=peak_names,
        nominal_centers=centers,
        center_tolerance_nm=float(config.gaussian_center_tolerance_nm),
        sigma_init=sigmas,
        sigma_bounds_nm=(float(config.gaussian_sigma_bounds_nm[0]), float(config.gaussian_sigma_bounds_nm[1])),
        positive_amplitudes=config.gaussian_positive_amplitudes,
    )



def _make_model_functions(spec: _GaussianModelSpec, xmin: float, xmax: float, w_seed: np.ndarray):
    n_peaks = len(spec.nominal_centers)

    def unpack(pars: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = 0
        if spec.baseline_mode == "none":
            baseline = np.array([0.0, 0.0], dtype=float)
        elif spec.baseline_mode == "constant":
            baseline = np.array([float(pars[idx]), 0.0], dtype=float)
            idx += 1
        else:  # linear
            baseline = np.array([float(pars[idx]), float(pars[idx + 1])], dtype=float)
            idx += 2

        amps = np.asarray(pars[idx : idx + n_peaks], dtype=float)
        idx += n_peaks

        if spec.center_mode == "fixed":
            centers = np.asarray(spec.nominal_centers, dtype=float)
        elif spec.center_mode == "shared_shift":
            delta = float(pars[idx])
            idx += 1
            centers = np.asarray(spec.nominal_centers, dtype=float) + delta
        else:
            centers = np.asarray(pars[idx : idx + n_peaks], dtype=float)
            idx += n_peaks

        if spec.width_mode == "shared":
            sigma_shared = float(pars[idx])
            idx += 1
            sigmas = np.full(n_peaks, sigma_shared, dtype=float)
        else:
            sigmas = np.asarray(pars[idx : idx + n_peaks], dtype=float)
            idx += n_peaks

        return baseline, amps, centers, sigmas

    def model(x: np.ndarray, *pars: float) -> np.ndarray:
        baseline, amps, centers, sigmas = unpack(pars)
        x = np.asarray(x, dtype=float)
        x_mid = float(np.mean(x))
        out = np.zeros_like(x, dtype=float)
        if spec.baseline_mode != "none":
            out += baseline[0] + baseline[1] * (x - x_mid)
        for amp, center, sigma in zip(amps, centers, sigmas, strict=False):
            out += _gaussian(x, float(amp), float(center), float(sigma))
        return out

    def evaluate(x: np.ndarray, pars: Sequence[float]) -> _FitEval:
        baseline, amps, centers, sigmas = unpack(pars)
        x = np.asarray(x, dtype=float)
        x_mid = float(np.mean(x))
        if spec.baseline_mode == "none":
            baseline_y = np.zeros_like(x, dtype=float)
        else:
            baseline_y = baseline[0] + baseline[1] * (x - x_mid)
        component_ys: list[np.ndarray] = []
        for amp, center, sigma in zip(amps, centers, sigmas, strict=False):
            component_ys.append(_gaussian(x, float(amp), float(center), float(sigma)))
        total_y = baseline_y.copy()
        for comp in component_ys:
            total_y += comp
        return _FitEval(
            total_y=total_y,
            baseline_y=baseline_y,
            component_ys=component_ys,
            centres_nm=[float(c) for c in centers],
            sigmas_nm=[float(s) for s in sigmas],
            amplitudes=[float(a) for a in amps],
        )

    def initial_guess_and_bounds() -> tuple[list[float], list[float], list[float]]:
        y_min = float(np.nanpercentile(y_int_global, 10))
        y_max = float(np.nanmax(y_int_global))
        amp0 = max(y_max - y_min, 1.0e-12)

        p0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []

        if spec.baseline_mode == "none":
            pass
        elif spec.baseline_mode == "constant":
            p0.append(y_min)
            lower.append(-np.inf)
            upper.append(np.inf)
        else:  # linear
            p0.extend([y_min, 0.0])
            lower.extend([-np.inf, -np.inf])
            upper.extend([np.inf, np.inf])

        amp_lower = 0.0 if spec.positive_amplitudes else -np.inf
        # Give each fixed/bounded component a local amplitude seed. This avoids
        # weak components near the end of the window starting essentially flat
        # when several peaks share a common sigma.
        for center in spec.nominal_centers:
            if w_seed.size >= 2 and y_int_global.size == w_seed.size and w_seed[0] <= center <= w_seed[-1]:
                local_amp0 = max(float(np.interp(center, w_seed, y_int_global)) - y_min, 0.05 * amp0)
            else:
                local_amp0 = amp0 / max(n_peaks, 1)
            p0.append(local_amp0)
            lower.append(amp_lower)
            upper.append(np.inf)

        if spec.center_mode == "fixed":
            pass
        elif spec.center_mode == "shared_shift":
            p0.append(0.0)
            lower.append(-abs(spec.center_tolerance_nm))
            upper.append(abs(spec.center_tolerance_nm))
        elif spec.center_mode == "bounded":
            for center in spec.nominal_centers:
                p0.append(center)
                lower.append(center - abs(spec.center_tolerance_nm))
                upper.append(center + abs(spec.center_tolerance_nm))
        else:  # free
            for center in spec.nominal_centers:
                p0.append(center)
                lower.append(xmin)
                upper.append(xmax)

        sigma_lo, sigma_hi = spec.sigma_bounds_nm
        if spec.width_mode == "shared":
            p0.append(float(np.mean(spec.sigma_init)))
            lower.append(sigma_lo)
            upper.append(sigma_hi)
        else:
            for sigma in spec.sigma_init:
                p0.append(float(sigma))
                lower.append(sigma_lo)
                upper.append(sigma_hi)

        return p0, lower, upper

    y_int_global = np.empty(0, dtype=float)

    def attach_y(y_int: np.ndarray) -> None:
        nonlocal y_int_global
        y_int_global = np.asarray(y_int, dtype=float)

    return model, evaluate, initial_guess_and_bounds, attach_y



def _integrate_gaussian_fit(w: np.ndarray, y: np.ndarray, xmin: float, xmax: float, config: IntegralConfig) -> IntegralResult:
    w_int, y_int = _hardcut_with_boundaries(w, y, xmin, xmax)
    if w_int.size < 5:
        return IntegralResult(np.nan, "gaussian_fit", xmin, xmax, int(w_int.size), {"reason": "not_enough_points"})

    spec = _build_gaussian_model_spec(config, w_int, y_int)
    model, evaluate, initial_guess_and_bounds, attach_y = _make_model_functions(spec, xmin, xmax, w_int)
    attach_y(y_int)
    p0, lower, upper = initial_guess_and_bounds()

    try:
        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(
            model,
            w_int,
            y_int,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50000,
        )
    except Exception as exc:
        fallback = _integrate_trapz(w, y, xmin, xmax, config)
        metadata = dict(fallback.metadata)
        metadata["fallback_reason"] = type(exc).__name__
        metadata["gaussian_fit_failed"] = True
        return IntegralResult(fallback.value, "gaussian_fit_fallback_trapz", xmin, xmax, fallback.n_points, metadata)

    fit_eval = evaluate(w_int, popt)

    component_integrals: list[float] = []
    value = 0.0
    for amp, mu, sigma in zip(fit_eval.amplitudes, fit_eval.centres_nm, fit_eval.sigmas_nm, strict=False):
        cdf_hi = _normal_cdf((xmax - mu) / sigma)
        cdf_lo = _normal_cdf((xmin - mu) / sigma)
        comp_val = float(amp * sigma * np.sqrt(2.0 * np.pi) * (cdf_hi - cdf_lo))
        component_integrals.append(comp_val)
        value += comp_val

    baseline_integral = 0.0
    if config.gaussian_integrate_baseline and spec.baseline_mode != "none":
        x_mid = float(np.mean(w_int))
        if spec.baseline_mode == "constant":
            b0 = float(popt[0])
            baseline_integral = b0 * (xmax - xmin)
        else:
            b0, b1 = float(popt[0]), float(popt[1])
            baseline_integral = b0 * (xmax - xmin) + 0.5 * b1 * ((xmax - x_mid) ** 2 - (xmin - x_mid) ** 2)
        value += baseline_integral

    x_fit = np.linspace(xmin, xmax, 1200)
    fit_dense = evaluate(x_fit, popt)

    metadata: dict[str, object] = {
        "n_gaussians": len(spec.nominal_centers),
        "gaussian_center_mode": spec.center_mode,
        "gaussian_width_mode": spec.width_mode,
        "gaussian_baseline_mode": spec.baseline_mode,
        "gaussian_peak_names": spec.peak_names,
        "component_integrals": component_integrals,
        "baseline_integral": baseline_integral,
        "curve_fit_covariance_shape": tuple(np.shape(pcov)),
    }
    for j, peak_name in enumerate(spec.peak_names):
        metadata[f"{peak_name}_amp"] = fit_eval.amplitudes[j]
        metadata[f"{peak_name}_mu_nm"] = fit_eval.centres_nm[j]
        metadata[f"{peak_name}_sigma_nm"] = fit_eval.sigmas_nm[j]
        metadata[f"{peak_name}_integral"] = component_integrals[j]

    plot_payload: dict[str, object] = {
        "window_x_nm": w_int,
        "window_y": y_int,
        "full_x_nm": w,
        "full_y": y,
        "fit_x_nm": x_fit,
        "fit_total_y": fit_dense.total_y,
        "fit_baseline_y": fit_dense.baseline_y,
        "fit_component_ys": fit_dense.component_ys,
        "peak_names": spec.peak_names,
        "component_integrals": component_integrals,
        "centres_nm": fit_dense.centres_nm,
        "sigmas_nm": fit_dense.sigmas_nm,
        "amplitudes": fit_dense.amplitudes,
        "xmin_nm": xmin,
        "xmax_nm": xmax,
        "integral_value": float(value),
        "baseline_integral": float(baseline_integral),
    }

    return IntegralResult(float(value), "gaussian_fit", xmin, xmax, int(w_int.size), metadata, plot_payload)
