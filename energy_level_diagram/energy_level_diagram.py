#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
energy_level_diagram.py

Plantilla sencilla para dibujar diagramas de niveles de energía con Matplotlib.

Uso:
    python energy_level_diagram.py

Salida:
    energy_level_diagram.pdf
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

OUTPUT_PDF = Path("energy_level_diagram.pdf")

FIGSIZE = (13, 7)
Y_MIN = -1.1
Y_MAX = 16.5

BLUE = "#3355aa"
GREEN = "darkgreen"
RED = "red"
NAVY = "navy"


def setup_axis():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.set_ylabel("Energy (eV)", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 17, 1))
    ax.tick_params(axis="y", direction="in", length=8, width=1.5, labelsize=13)

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_linewidth(1.5)

    return fig, ax


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def level(ax, x, y, label=None, width=0.55, color="black", lw=3, dy=0.15):
    """
    Dibuja un nivel horizontal en la posición x y energía y.
    """
    ax.hlines(y, x - width / 2, x + width / 2, color=color, lw=lw)

    if label is not None:
        ax.text(
            x,
            y + dy,
            label,
            ha="center",
            va="bottom",
            fontsize=13,
        )


def manifold(ax, x, y_min, y_max, n=12, width=0.55, color="red", lw=1.5, label=None):
    """
    Dibuja un conjunto de muchos niveles próximos.
    """
    ys = np.linspace(y_min, y_max, n)

    for y in ys:
        ax.hlines(y, x - width / 2, x + width / 2, color=color, lw=lw)

    if label is not None:
        ax.text(
            x,
            y_max + 0.25,
            label,
            ha="center",
            va="bottom",
            fontsize=13,
        )


def transition(
    ax,
    x1,
    y1,
    x2,
    y2,
    label=None,
    style="-",
    color="black",
    lw=1.8,
    text_offset=(0, 0),
):
    """
    Dibuja una flecha entre dos estados.
    """
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            lw=lw,
            color=color,
            linestyle=style,
            shrinkA=2,
            shrinkB=2,
        ),
    )

    if label is not None:
        xm = 0.5 * (x1 + x2) + text_offset[0]
        ym = 0.5 * (y1 + y2) + text_offset[1]

        ax.text(
            xm,
            ym,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
            ),
        )


def column_label(ax, x, text):
    """
    Etiqueta inferior para cada especie.
    """
    ax.text(
        x,
        -0.65,
        text,
        ha="center",
        va="top",
        fontsize=14,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="square,pad=0.25",
        ),
        clip_on=False,
    )


# ============================================================
# DIAGRAMA
# ============================================================

def draw_diagram():
    fig, ax = setup_axis()

    # Posiciones horizontales ficticias.
    # Cambia estos valores si quieres separar más o menos las columnas.
    x_xe_liq = 1.0
    x_ar_liq = 2.6
    x_n2 = 5.0
    x_ar_gas = 7.2
    x_xe_gas = 8.8

    # --------------------------------------------------------
    # Xe in liquid Ar
    # --------------------------------------------------------

    level(ax, x_xe_liq, 10.7, r"$\mathrm{Xe}^+$", color=BLUE)
    level(ax, x_xe_liq, 10.1, r"$\mathrm{Xe}^*(n=1,\ ^2P_{1/2})$", color=GREEN)
    level(ax, x_xe_liq, 9.8,  r"$\mathrm{Xe}^*(n=2,\ ^2P_{3/2})$", color=GREEN)
    level(ax, x_xe_liq, 8.9,  r"$\mathrm{Xe}^*(n=1,\ ^2P_{3/2})$", color=GREEN)

    manifold(
        ax,
        x_xe_liq,
        7.0,
        7.45,
        n=5,
        color=NAVY,
        label=r"$\mathrm{Xe}_2^*(^{1,3}\Sigma_u^+)$",
    )

    level(ax, x_xe_liq, 0.0, "Ground state", color=GREEN)

    transition(ax, x_xe_liq, 9.8, x_xe_liq, 7.45, style="--")
    transition(ax, x_xe_liq, 7.0, x_xe_liq, 0.4, label="175±10 nm")

    column_label(ax, x_xe_liq, "Xe in liquid Ar")

    # --------------------------------------------------------
    # Liquid Ar
    # --------------------------------------------------------

    level(ax, x_ar_liq, 13.9, r"$\mathrm{Ar}^+$ conduction band", color=BLUE, width=0.7)
    level(ax, x_ar_liq, 12.2, "Excitons", color=GREEN)
    level(ax, x_ar_liq, 12.0, r"$\mathrm{Ar}^*(n=1,\ ^2P_{1/2})$", color=GREEN)
    level(ax, x_ar_liq, 11.8, r"$\mathrm{Ar}^*(n=1,\ ^2P_{3/2})$", color=GREEN)

    manifold(
        ax,
        x_ar_liq,
        9.2,
        10.0,
        n=6,
        color=NAVY,
        label=r"$\mathrm{Ar}_2^*(^{1,3}\Sigma_u^+)$",
    )

    level(ax, x_ar_liq, 0.0, "Valence band", color=GREEN)

    transition(ax, x_ar_liq, 12.0, x_ar_liq, 10.0, style="--")
    transition(ax, x_ar_liq, 9.2, x_ar_liq, 0.5, label="128±10 nm")

    column_label(ax, x_ar_liq, "Liquid Ar")

    # --------------------------------------------------------
    # Gaseous N2
    # --------------------------------------------------------

    level(ax, x_n2, 15.6, r"$\mathrm{N}_2^+$", color=BLUE)

    manifold(ax, x_n2 - 0.35, 6.2, 9.8, n=25, color=RED, label="N+N")
    manifold(ax, x_n2 + 0.35, 9.0, 12.0, n=18, color=RED, label="N+N")

    level(ax, x_n2, 0.0, r"$\mathrm{N}_2(X^1\Sigma_g^+)$", color=RED)

    transition(
        ax,
        x_n2 + 0.35,
        12.0,
        x_n2 + 0.35,
        8.0,
        label="2nd pos. sys.\n310-440 nm",
        text_offset=(0.65, 0),
    )

    transition(
        ax,
        x_n2 - 0.35,
        7.0,
        x_n2 - 0.35,
        0.5,
        label="1st pos. sys.\n500-2500 nm",
        text_offset=(0.65, 0),
    )

    column_label(ax, x_n2, r"Gaseous N$_2$")

    # --------------------------------------------------------
    # Gaseous Ar
    # --------------------------------------------------------

    level(ax, x_ar_gas, 15.8, r"$\mathrm{Ar}^+$", color=BLUE)

    manifold(
        ax,
        x_ar_gas,
        13.0,
        13.45,
        n=5,
        color=RED,
        label=r"$\mathrm{Ar}^*(3p^54p^1)$",
    )

    manifold(
        ax,
        x_ar_gas,
        11.55,
        11.85,
        n=4,
        color=RED,
        label=r"$\mathrm{Ar}^*(3p^54s^1)$",
    )

    manifold(
        ax,
        x_ar_gas,
        9.35,
        10.0,
        n=6,
        color=NAVY,
        label=r"$\mathrm{Ar}_2^*(^{1,3}\Sigma_u^+)$",
    )

    level(ax, x_ar_gas, 0.0, r"$\mathrm{Ar}(3p^6)$", color=RED)

    transition(ax, x_ar_gas, 13.2, x_ar_gas, 11.85, label="690-850 nm")
    transition(ax, x_ar_gas, 11.55, x_ar_gas, 10.0, style="--")
    transition(ax, x_ar_gas, 9.35, x_ar_gas, 0.5, label="128±12 nm")

    column_label(ax, x_ar_gas, "Gaseous Ar")

    # --------------------------------------------------------
    # Gaseous Xe
    # --------------------------------------------------------

    level(ax, x_xe_gas, 12.1, r"$\mathrm{Xe}^+$", color=BLUE)

    manifold(
        ax,
        x_xe_gas,
        8.0,
        9.55,
        n=5,
        color=RED,
        label=r"$\mathrm{Xe}^*(5p^56s^1)$",
    )

    manifold(
        ax,
        x_xe_gas,
        7.0,
        7.45,
        n=5,
        color=NAVY,
        label=r"$\mathrm{Xe}_2^*(^{1,3}\Sigma_u^+)$",
    )

    level(ax, x_xe_gas, 0.0, r"$\mathrm{Xe}(5p^6)$", color=RED)

    transition(ax, x_xe_gas, 8.0, x_xe_gas, 7.45, style="--")
    transition(ax, x_xe_gas, 7.0, x_xe_gas, 0.5, label="175±20 nm")

    column_label(ax, x_xe_gas, "Gaseous Xe")

    # --------------------------------------------------------
    # Transferencias entre columnas
    # --------------------------------------------------------

    transition(ax, x_xe_liq + 0.3, 9.8, x_ar_liq - 0.3, 9.8, style="--")
    transition(ax, x_ar_liq + 0.3, 9.4, x_n2 - 0.6, 9.4, style="--")
    transition(ax, x_ar_gas - 0.35, 9.6, x_n2 + 0.55, 9.6, style="--")
    transition(ax, x_ar_gas + 0.35, 9.6, x_xe_gas - 0.35, 9.6, style="--")

    return fig, ax


def main():
    fig, ax = draw_diagram()
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.show()
    print(f"Saved: {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
