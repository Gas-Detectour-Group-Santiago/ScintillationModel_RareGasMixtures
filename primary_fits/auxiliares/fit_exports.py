from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_num(x: float) -> str:
    if x is None or not np.isfinite(x):
        return r"--"
    return rf"\num{{{float(x):.3g}}}"


def _fmt_asym(m: float, p: float) -> str:
    if (not np.isfinite(m)) or (not np.isfinite(p)):
        return r"--"
    if abs(m) == 0 and abs(p) == 0:
        return r"--"
    return rf"$^{{+{_fmt_num(p)}}}_{{-{_fmt_num(m)}}}$"


def build_parameter_summary(
    names: list[str],
    tex_names: list[str],
    central: np.ndarray,
    central_err: np.ndarray,
    stat_minus: np.ndarray,
    stat_plus: np.ndarray,
    syst_minus: np.ndarray,
    syst_plus: np.ndarray,
    fixed_idx: list[int],
) -> pd.DataFrame:
    central = np.asarray(central, dtype=float)
    total_minus = np.sqrt(np.asarray(stat_minus) ** 2 + np.asarray(syst_minus) ** 2)
    total_plus = np.sqrt(np.asarray(stat_plus) ** 2 + np.asarray(syst_plus) ** 2)

    return pd.DataFrame(
        {
            "name": names,
            "tex_name": tex_names,
            "value": central,
            "fit_uncertainty": central_err,
            "stat_minus": stat_minus,
            "stat_plus": stat_plus,
            "syst_minus": syst_minus,
            "syst_plus": syst_plus,
            "total_minus": total_minus,
            "total_plus": total_plus,
            "fixed": [i in set(fixed_idx) for i in range(len(names))],
        }
    )


def export_stat_syst_latex(
    summary: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Parámetro & Valor & Stat. & Syst. & Total \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{row['tex_name']} & {_fmt_num(row['value'])} & "
            f"{_fmt_asym(row['stat_minus'], row['stat_plus'])} & "
            f"{_fmt_asym(row['syst_minus'], row['syst_plus'])} & "
            f"{_fmt_asym(row['total_minus'], row['total_plus'])} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_toys(path: Path, names: list[str], toys: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if toys.size == 0:
        pd.DataFrame(columns=["toy_id"] + names).to_csv(path, index=False)
        return
    df = pd.DataFrame(toys, columns=names)
    df.insert(0, "toy_id", np.arange(len(df), dtype=int))
    df.to_csv(path, index=False)


def export_vector(path: Path, names: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": names, "value": values}).to_csv(path, index=False)


def export_matrix(path: Path, names: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path)

def toy_covariance_correlation(toys: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Compute empirical covariance and correlation matrices from toy-fit parameters.

    The returned matrices keep all parameters. Parameters that do not vary across
    the valid toy ensemble (typically fixed parameters) have undefined
    correlations and are therefore left as NaN outside the diagonal.
    """
    names = list(names)
    empty = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)

    toys = np.asarray(toys, dtype=float)
    if toys.size == 0 or toys.ndim != 2 or toys.shape[1] != len(names):
        return empty.copy(), empty.copy(), 0

    df = pd.DataFrame(toys, columns=names)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    n_valid = int(len(df))
    if n_valid < 2:
        return empty.copy(), empty.copy(), n_valid

    cov = df.cov()
    corr = df.corr()

    # Make the diagonal explicit for parameters with a finite variance. For fixed
    # or numerically constant parameters the variance is zero and the correlation
    # is not defined; keeping NaN avoids creating artificial correlations.
    variances = np.diag(cov.to_numpy(dtype=float))
    corr_values = corr.to_numpy(dtype=float)
    for i, var in enumerate(variances):
        if np.isfinite(var) and var > 0:
            corr_values[i, i] = 1.0
        else:
            corr_values[i, i] = np.nan
    corr = pd.DataFrame(corr_values, index=names, columns=names)

    return cov, corr, n_valid


def export_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def plot_parameter_correlation(
    path: Path,
    corr: pd.DataFrame,
    labels: list[str] | None = None,
    title: str = "Parameter correlations",
) -> None:
    """Export a seaborn PDF heatmap for a parameter correlation matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="white",
        font="serif",
        rc={
            "axes.grid": False,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        },
    )

    values = corr.to_numpy(dtype=float)
    n = len(corr.columns)
    if labels is None:
        labels = list(corr.columns)

    mask = ~np.isfinite(values)
    plot_df = pd.DataFrame(values, index=labels, columns=labels)
    annot = np.empty(values.shape, dtype=object)
    annot[:] = ""
    finite = np.isfinite(values)
    annot[finite] = np.vectorize(lambda x: f"{x:.2f}")(values[finite])
    diag_nan = np.eye(n, dtype=bool) & mask
    annot[diag_nan] = "--"

    fig_size = max(5.0, 0.58 * n + 2.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        plot_df,
        ax=ax,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        cmap="vlag",
        mask=mask & ~diag_nan,
        square=True,
        linewidths=0.45,
        linecolor="white",
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": r"Coeficiente de correlación $\rho_{ij}$", "shrink": 0.82},
    )
    ax.set_title(title, pad=10)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels, rotation=0)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

