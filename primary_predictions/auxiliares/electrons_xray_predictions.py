from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator


PROJECT_ROOT_FROM_MODULE = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FROM_MODULE) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_MODULE))

from plot_style import (  # noqa: E402
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    LEGEND,
    LINEWIDTH_MAIN,
    MARKERSIZE,
    apply_axis_style,
    palette,
    setup_style,
)


DEFAULT_NORMALIZATION = "ArCF4"

POPULATION_COLUMNS: tuple[str, ...] = (
    "CF4",
    "CF3",
    "N2_star",
    "Ar_3rd",
    "Ar_dbleStar",
    "Ar_1s4_1s5",
    "Ar_1s2_1s3",
    "Ar_2nd_precursor",
)


@dataclass(frozen=True)
class EnergyChannel:
    id: str
    label: str
    gas_mixture: str
    color_index: int


CHANNELS: tuple[EnergyChannel, ...] = (
    EnergyChannel("N2_UV", r"N$_2$ UV", "N2", 0),
    EnergyChannel("CF4_VIS", r"CF$_4$ VIS", "CF4", 1),
    EnergyChannel("Ar2nd_99_1", r"Ar 2nd (99/1)", "ArCF4", 2),
    EnergyChannel("Ar2nd_pure", r"Ar 2nd (pure Ar)", "Ar", 3),
)


@dataclass(frozen=True)
class WexcGas:
    id: str
    label: str
    gas_mixture: str
    color_index: int


WEXC_GASES: tuple[WexcGas, ...] = (
    WexcGas("Ar", "Ar", "Ar", 0),
    WexcGas("Xe", "Xe", "Xe", 1),
    WexcGas("CF4", r"CF$_4$", "CF4", 2),
    WexcGas("N2", r"N$_2$", "N2", 3),
)

PARTICLE_STYLE = {
    "xray": {"label": "X-ray", "marker": "o", "linestyle": "-"},
    "electron": {"label": "Electron", "marker": "s", "linestyle": "--"},
}


def _configure_plot_style() -> None:
    setup_style(grid=False, use_latex=False, context="single")


def _channel_colors() -> dict[str, np.ndarray]:
    colors = palette(len(CHANNELS), start=0.10, stop=0.90)
    return {channel.id: colors[channel.color_index] for channel in CHANNELS}


def _wexc_colors() -> dict[str, np.ndarray]:
    colors = palette(len(WEXC_GASES), start=0.10, stop=0.90)
    return {gas.id: colors[gas.color_index] for gas in WEXC_GASES}


def _boxed_legend_kwargs(**overrides: object) -> dict[str, object]:
    kwargs = LEGEND.as_kwargs(
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="0.35",
        fancybox=True,
    )
    kwargs.update(overrides)
    return kwargs


def _load_inputs(project_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, object]]:
    project_root = Path(project_root)
    for folder in (project_root / "models", project_root / "primary_predictions"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))

    population_csv = project_root / "data" / "Primary_DegradData" / "electrons_xRay_energy_cases.csv"
    required_summary_columns = {
        "total_excitations",
        "Errtotal_excitations",
        "Wexc_eV",
        "ErrWexc_eV",
    }
    regenerate = not population_csv.exists()
    if not regenerate:
        regenerate = not required_summary_columns.issubset(
            set(pd.read_csv(population_csv, nrows=1).columns)
        )
    if regenerate:
        data_dir = project_root / "data"
        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))
        from Analysis_primary_degrad import analyse_electrons_xray_energy_cases

        analyse_electrons_xray_energy_cases()
    cases = pd.read_csv(population_csv)

    from primary_predictions.auxiliares.fit_products import FitProductStore
    from Ar2nd_continium import read_ar2nd_parameters

    store = FitProductStore(project_root)
    arcf4 = store.load("ArCF4_primary").central
    arn2 = store.load("ArN2_primary").central
    ar2 = read_ar2nd_parameters(project_root / "data" / "Parameters" / "Ar2nd_continium.csv")
    return cases, arcf4, arn2, ar2


def _single_row(row: pd.Series, concentration: float) -> pd.DataFrame:
    out = pd.DataFrame([row.to_dict()])
    out["concentration"] = float(concentration)
    return out


def _evaluate_channel(
    channel_id: str,
    row: pd.Series,
    arcf4_params: np.ndarray,
    arn2_params: np.ndarray,
    ar2_params: dict[str, object],
    *,
    normalization: str = DEFAULT_NORMALIZATION,
) -> float:
    energy_kev = max(float(row["energy_kev"]), 1.0e-30)
    pressure = float(row.get("pressure_bar", 1.0))

    if channel_id == "N2_UV":
        n_norm, p_n2, tau_n2, k_n2_q_n2 = map(float, arn2_params[:4])
        inv_tau = 1.0 / max(tau_n2, 1.0e-30)
        survival = inv_tau / (inv_tau + pressure * max(k_n2_q_n2, 0.0))
        raw = n_norm * survival * float(row["N2_star"]) * p_n2 / energy_kev
        if normalization != "ArCF4":
            raise ValueError(
                f"Normalización no soportada para estas figuras: {normalization!r}. "
                "La convención por defecto es ArCF4."
            )
        return raw * 1000.0 / float(arcf4_params[0])

    if channel_id == "CF4_VIS":
        from ArCF4 import energy_X_ray_CF4, theory_yield_vis

        raw_reference_energy = float(
            np.ravel(theory_yield_vis(arcf4_params, _single_row(row, 1.0), 1.0, pressure))[0]
        )
        corrected = raw_reference_energy * float(energy_X_ray_CF4) / energy_kev
        return corrected * 1000.0 / float(arcf4_params[0])

    if channel_id in {"Ar2nd_99_1", "Ar2nd_pure"}:
        from Ar2nd_continium import theory_yield_ar2nd_continium

        if channel_id == "Ar2nd_99_1":
            fraction = 0.01
            gas_mixture = "ArCF4"
            additive = "CF4"
        else:
            fraction = 0.0
            gas_mixture = "Ar"
            additive = None

        # The second-continuum model is already absolute: it converts the
        # simulated precursor populations directly into photons per keV.  It
        # therefore shares the Ar-CF4 comparison convention but is not divided
        # by a fitted optical Nnorm.
        raw_per_kev = theory_yield_ar2nd_continium(
            ar2_params,
            _single_row(row, fraction),
            fraction,
            pressure,
            gas_mixture=gas_mixture,
            additive=additive,
            energy_xray_ev=energy_kev,
        )
        return float(raw_per_kev) * 1000.0

    raise KeyError(f"Canal electron/X-ray no definido: {channel_id}")


def build_simulated_yields(project_root: Path, *, normalization: str = DEFAULT_NORMALIZATION) -> pd.DataFrame:
    cases, arcf4, arn2, ar2 = _load_inputs(project_root)
    rows: list[dict[str, object]] = []
    for channel in CHANNELS:
        selected = cases.loc[cases["gas_mixture"].astype(str) == channel.gas_mixture].copy()
        for _, case in selected.iterrows():
            rows.append(
                {
                    "channel": channel.id,
                    "channel_label": channel.label,
                    "particle": str(case["particle"]),
                    "energy_kev": float(case["energy_kev"]),
                    "yield_ph_MeV": _evaluate_channel(channel.id, case, arcf4, arn2, ar2, normalization=normalization),
                    "gas_mixture": str(case["gas_mixture"]),
                    "pressure_bar": float(case.get("pressure_bar", 1.0)),
                    "electric_field_v_cm_bar": float(case.get("electric_field_v_cm_bar", np.nan)),
                    "source_txt": str(case.get("source_txt", "")),
                    "normalization": normalization,
                }
            )
    return pd.DataFrame(rows).sort_values(["channel", "particle", "energy_kev"]).reset_index(drop=True)


def _interpolate_population_rows(rows: pd.DataFrame, n_points: int = 240) -> pd.DataFrame:
    rows = rows.sort_values("energy_kev").drop_duplicates("energy_kev", keep="first")
    if len(rows) < 2:
        return pd.DataFrame()

    energies = rows["energy_kev"].to_numpy(dtype=float)
    log_energy = np.log10(energies)
    grid_energy = np.logspace(log_energy.min(), log_energy.max(), n_points)
    grid_log = np.log10(grid_energy)

    dense = pd.DataFrame({"energy_kev": grid_energy})
    for column in POPULATION_COLUMNS:
        values = pd.to_numeric(rows.get(column, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        # Degrad populations scale approximately with deposited energy.
        # Interpolate the population density per keV and reconstruct the total
        # population afterwards; direct interpolation of raw populations in
        # log(E) produces an artificial hump between 12 keV and 1.5 MeV.
        values_per_kev = np.divide(values, energies, out=np.zeros_like(values), where=energies > 0.0)
        if len(rows) >= 3:
            interpolated_per_kev = PchipInterpolator(
                log_energy, values_per_kev, extrapolate=False
            )(grid_log)
        else:
            interpolated_per_kev = np.interp(grid_log, log_energy, values_per_kev)
        dense[column] = np.clip(interpolated_per_kev * grid_energy, 0.0, None)

    for column in (
        "particle",
        "gas_mixture",
        "gas_label",
        "additive",
        "additive_fraction",
        "pressure_bar",
        "electric_field_v_cm_bar",
    ):
        dense[column] = rows.iloc[0].get(column, "")
    return dense


def build_interpolated_yields(project_root: Path, *, normalization: str = DEFAULT_NORMALIZATION) -> pd.DataFrame:
    cases, arcf4, arn2, ar2 = _load_inputs(project_root)
    output: list[dict[str, object]] = []
    for channel in CHANNELS:
        channel_cases = cases.loc[cases["gas_mixture"].astype(str) == channel.gas_mixture]
        for particle, rows in channel_cases.groupby("particle", sort=False):
            dense = _interpolate_population_rows(rows)
            if dense.empty:
                continue
            for _, row in dense.iterrows():
                output.append(
                    {
                        "channel": channel.id,
                        "channel_label": channel.label,
                        "particle": str(particle),
                        "energy_kev": float(row["energy_kev"]),
                        "yield_ph_MeV": _evaluate_channel(channel.id, row, arcf4, arn2, ar2, normalization=normalization),
                        "normalization": normalization,
                    }
                )
    return pd.DataFrame(output)


def _series_handles() -> tuple[list[Line2D], list[Line2D]]:
    colors = _channel_colors()
    channel_handles = [
        Line2D([0], [0], color=colors[channel.id], lw=LINEWIDTH_MAIN, label=channel.label)
        for channel in CHANNELS
    ]
    particle_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=5,
            label=style["label"],
        )
        for style in PARTICLE_STYLE.values()
    ]
    return channel_handles, particle_handles


def _combined_series_handles() -> list[Line2D]:
    channel_handles, particle_handles = _series_handles()
    spacer = Line2D([], [], linestyle="none", marker=None, color="none", label="Incident type")
    return [*channel_handles, spacer, *particle_handles]


def _plot_absolute(simulated: pd.DataFrame, dense: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = _channel_colors()
    for channel in CHANNELS:
        color = colors[channel.id]
        for particle, style in PARTICLE_STYLE.items():
            points = simulated.loc[
                (simulated["channel"] == channel.id) & (simulated["particle"] == particle)
            ].sort_values("energy_kev")
            curve = dense.loc[(dense["channel"] == channel.id) & (dense["particle"] == particle)]
            if not curve.empty:
                ax.plot(
                    curve["energy_kev"],
                    curve["yield_ph_MeV"],
                    color=color,
                    linestyle=style["linestyle"],
                    linewidth=LINEWIDTH_MAIN,
                )
            if not points.empty:
                ax.plot(
                    points["energy_kev"],
                    points["yield_ph_MeV"],
                    linestyle="none",
                    marker=style["marker"],
                    color=color,
                    markersize=MARKERSIZE + 0.5,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Incident energy [keV]")
    ax.set_ylabel(r"Primary yield [ph MeV$^{-1}$]")
    combined_handles = _combined_series_handles()
    ax.legend(
        handles=combined_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        bbox_transform=ax.transAxes,
        **_boxed_legend_kwargs(handlelength=2.1, fontsize=9.4),
    )
    apply_axis_style(ax)
    ax.margins(x=0.05, y=0.12)
    fig.savefig(output)
    plt.close(fig)


def _reference_values(simulated: pd.DataFrame) -> dict[str, float]:
    refs: dict[str, float] = {}
    for channel in CHANNELS:
        rows = simulated.loc[
            (simulated["channel"] == channel.id)
            & (simulated["particle"] == "xray")
            & np.isclose(simulated["energy_kev"], 12.0)
        ]
        if not rows.empty:
            refs[channel.id] = float(rows.iloc[0]["yield_ph_MeV"])
    return refs


def _plot_relative(simulated: pd.DataFrame, dense: pd.DataFrame, output: Path) -> None:
    refs = _reference_values(simulated)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = _channel_colors()
    for channel in CHANNELS:
        reference = refs.get(channel.id)
        if reference is None or reference == 0.0:
            continue
        color = colors[channel.id]
        for particle, style in PARTICLE_STYLE.items():
            points = simulated.loc[
                (simulated["channel"] == channel.id) & (simulated["particle"] == particle)
            ].sort_values("energy_kev")
            curve = dense.loc[(dense["channel"] == channel.id) & (dense["particle"] == particle)]
            if not curve.empty:
                ax.plot(
                    curve["energy_kev"],
                    curve["yield_ph_MeV"] / reference,
                    color=color,
                    linestyle=style["linestyle"],
                    linewidth=LINEWIDTH_MAIN,
                )
            if not points.empty:
                ax.plot(
                    points["energy_kev"],
                    points["yield_ph_MeV"] / reference,
                    linestyle="none",
                    marker=style["marker"],
                    color=color,
                    markersize=MARKERSIZE + 0.5,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                )
    ax.axhline(1.0, color="0.65", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Incident energy [keV]")
    ax.set_ylabel(r"$Y(E)/Y_{\mathrm{X\ ray}}(12\,\mathrm{keV})$")
    combined_handles = _combined_series_handles()
    ax.legend(
        handles=combined_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        bbox_transform=ax.transAxes,
        **_boxed_legend_kwargs(fontsize=8.8, handlelength=2.0),
    )
    apply_axis_style(ax)
    ax.margins(x=0.05, y=0.12)
    fig.savefig(output)
    plt.close(fig)


def _plot_individual(simulated: pd.DataFrame, dense: pd.DataFrame, channel: EnergyChannel, output: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    color = _channel_colors()[channel.id]
    for particle, style in PARTICLE_STYLE.items():
        points = simulated.loc[
            (simulated["channel"] == channel.id) & (simulated["particle"] == particle)
        ].sort_values("energy_kev")
        curve = dense.loc[(dense["channel"] == channel.id) & (dense["particle"] == particle)]
        if not curve.empty:
            ax.plot(
                curve["energy_kev"],
                curve["yield_ph_MeV"],
                color=color,
                linestyle=style["linestyle"],
                linewidth=LINEWIDTH_MAIN,
                label=style["label"],
            )
        elif not points.empty:
            ax.plot([], [], color=color, linestyle=style["linestyle"], marker=style["marker"], label=style["label"])
        if not points.empty:
            ax.plot(
                points["energy_kev"],
                points["yield_ph_MeV"],
                linestyle="none",
                marker=style["marker"],
                color=color,
                markersize=MARKERSIZE + 0.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Incident energy [keV]")
    ax.set_ylabel(r"Primary yield [ph MeV$^{-1}$]")
    ax.set_title(channel.label)
    apply_axis_style(
        ax,
        legend=True,
        legend_kwargs=_boxed_legend_kwargs(loc="best", handlelength=2.1),
    )
    ax.margins(x=0.07, y=0.15)
    fig.savefig(output)
    plt.close(fig)


def build_wexc_points(project_root: Path) -> pd.DataFrame:
    cases, _, _, _ = _load_inputs(project_root)
    allowed = {gas.gas_mixture for gas in WEXC_GASES}
    columns = [
        "gas_mixture",
        "gas_label",
        "particle",
        "energy_kev",
        "total_excitations",
        "Errtotal_excitations",
        "Wexc_eV",
        "ErrWexc_eV",
        "pressure_bar",
        "electric_field_v_cm_bar",
        "source_txt",
    ]
    out = cases.loc[cases["gas_mixture"].isin(allowed), columns].copy()
    labels = {gas.gas_mixture: gas.label for gas in WEXC_GASES}
    out["gas_label"] = out["gas_mixture"].map(labels)
    return out.sort_values(["gas_mixture", "particle", "energy_kev"]).reset_index(drop=True)


def _wexc_handles() -> tuple[list[Line2D], list[Line2D]]:
    colors = _wexc_colors()
    gas_handles = [
        Line2D([0], [0], color=colors[gas.id], lw=LINEWIDTH_MAIN, label=gas.label)
        for gas in WEXC_GASES
    ]
    particle_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=5,
            label=style["label"],
        )
        for style in PARTICLE_STYLE.values()
    ]
    return gas_handles, particle_handles


def _combined_wexc_handles() -> list[Line2D]:
    gas_handles, particle_handles = _wexc_handles()
    spacer = Line2D([], [], linestyle="none", marker=None, color="none", label="Incident type")
    return [*gas_handles, spacer, *particle_handles]


def _plot_wexc(wexc: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = _wexc_colors()
    for gas in WEXC_GASES:
        color = colors[gas.id]
        for particle, style in PARTICLE_STYLE.items():
            points = wexc.loc[
                (wexc["gas_mixture"] == gas.gas_mixture)
                & (wexc["particle"] == particle)
            ].sort_values("energy_kev")
            if points.empty:
                continue
            ax.errorbar(
                points["energy_kev"],
                points["Wexc_eV"],
                yerr=points["ErrWexc_eV"],
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=LINEWIDTH_MAIN,
                markersize=MARKERSIZE + 0.5,
                markeredgecolor="white",
                markeredgewidth=0.45,
                capsize=2.5,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Incident energy [keV]")
    ax.set_ylabel(r"$W_{\mathrm{exc}}$ [eV]")
    combined_handles = _combined_wexc_handles()
    ax.legend(
        handles=combined_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        bbox_transform=ax.transAxes,
        **_boxed_legend_kwargs(handlelength=2.1, fontsize=9.4),
    )
    apply_axis_style(ax)
    ax.margins(x=0.05, y=0.10)
    fig.savefig(output)
    plt.close(fig)


def export_electrons_xray_predictions(
    project_root: Path,
    *,
    make_plots: bool = True,
    normalization: str = DEFAULT_NORMALIZATION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(project_root)
    _configure_plot_style()
    simulated = build_simulated_yields(project_root, normalization=normalization)
    dense = build_interpolated_yields(project_root, normalization=normalization)
    wexc = build_wexc_points(project_root)

    predictions_dir = project_root / "data" / "Predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    simulated_path = predictions_dir / "electrons_xRay_primary_yields.csv"
    dense_path = predictions_dir / "electrons_xRay_primary_yields_interpolated.csv"
    wexc_path = predictions_dir / "electrons_xRay_Wexc_pure_gases.csv"
    simulated.to_csv(simulated_path, index=False)
    dense.to_csv(dense_path, index=False)
    wexc.to_csv(wexc_path, index=False)

    if make_plots:
        plot_dir = project_root / "primary_predictions" / "plots" / "electrons_xRay"
        plot_dir.mkdir(parents=True, exist_ok=True)
        legacy_third_continuum = plot_dir / "primary_yield_electrons_xRay_Ar_3rd.pdf"
        if legacy_third_continuum.exists():
            legacy_third_continuum.unlink()

        _plot_absolute(simulated, dense, plot_dir / "primary_yield_electrons_xRay_all.pdf")
        _plot_relative(simulated, dense, plot_dir / "primary_yield_electrons_xRay_relative.pdf")
        _plot_wexc(wexc, plot_dir / "primary_Wexc_electrons_xRay_pure_gases.pdf")
        for channel in CHANNELS:
            _plot_individual(
                simulated,
                dense,
                channel,
                plot_dir / f"primary_yield_electrons_xRay_{channel.id}.pdf",
            )
        print(f"[primary_predictions] electron/X-ray plots: {plot_dir}")

    print(f"[primary_predictions] electron/X-ray normalization: {normalization}")
    print(f"[primary_predictions] electron/X-ray points: {simulated_path}")
    print(f"[primary_predictions] electron/X-ray interpolation: {dense_path}")
    print(f"[primary_predictions] electron/X-ray Wexc: {wexc_path}")
    return simulated, dense
