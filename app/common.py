from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scintillation.gui.style_config import temporary_style


def style_preview(style: dict[str, Any]) -> plt.Figure:
    """Preview every plot-style control used by the production renderers."""
    with temporary_style(style):
        x = np.linspace(0.15, 10.0, 30)
        figure, axis = plt.subplots(figsize=tuple(style["figure_wide"]))
        cmap = plt.get_cmap(style["cmap"])
        y_main = 2.0 + x + 0.35 * np.sin(x)
        y_secondary = 3.0 + 0.75 * x + 0.25 * np.cos(1.3 * x)
        axis.fill_between(
            x,
            y_main - 0.8,
            y_main + 0.8,
            color=cmap(0.50),
            alpha=float(style["band_alpha"]),
            label="Total band",
        )
        axis.plot(
            x,
            y_main,
            color=cmap(0.22),
            lw=float(style["line_width_main"]),
            label="Main model",
        )
        axis.plot(
            x,
            y_secondary,
            color=cmap(0.75),
            lw=float(style["line_width_secondary"]),
            linestyle="--",
            label="Secondary model",
        )
        sample = np.arange(2, len(x), 5)
        axis.errorbar(
            x[sample],
            y_main[sample],
            yerr=0.55,
            fmt="o",
            color=cmap(0.22),
            ms=float(style["marker_size"]),
            capsize=float(style["cap_size"]),
            elinewidth=float(style["errorbar_line_width"]),
            capthick=float(style["cap_thickness"]),
            label="Experimental data",
        )
        axis.minorticks_on()
        axis.tick_params(
            which="major",
            width=float(style["tick_major_width"]),
            length=float(style["tick_major_length"]),
            direction=str(style["tick_direction"]),
            top=bool(style["ticks_top"]),
            right=bool(style["ticks_right"]),
        )
        axis.tick_params(
            which="minor",
            width=float(style["tick_minor_width"]),
            length=float(style["tick_minor_length"]),
            direction=str(style["tick_direction"]),
            top=bool(style["ticks_top"]),
            right=bool(style["ticks_right"]),
        )
        for spine in axis.spines.values():
            spine.set_linewidth(float(style["axes_line_width"]))
        axis.set(xlabel="Reduced field E/p", ylabel="Scintillation yield", title="Style preview")
        axis.legend(ncol=int(style.get("legend_columns", 1)), title="Legend title")
        figure.tight_layout()
    return figure
