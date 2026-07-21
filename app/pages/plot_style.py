from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from app.common import style_preview
from app.context import PROJECT_ROOT
from scintillation.gui.style_config import (
    active_style_name,
    available_styles,
    load_style,
    save_style,
    set_active_style,
)


def page() -> None:
    st.title("Global plot style")
    st.caption("Edit a reusable preset. The active preset is used by the GUI and by the normal run scripts.")

    styles = available_styles(PROJECT_ROOT)
    if not styles:
        st.error("No style presets were found in config/styles.")
        return
    names = list(styles)
    active = active_style_name(PROJECT_ROOT)
    index = names.index(active) if active in names else 0
    selected = st.selectbox("Preset to edit", names, index=index)
    base = load_style(PROJECT_ROOT, selected)
    st.info(f"Active default: **{active}**")

    st.subheader("Typography and colour")
    c1, c2, c3 = st.columns(3)
    label = c1.text_input("Display name", value=str(base["label"]))
    font_family = c2.selectbox("Font family", ["serif", "sans-serif", "monospace"], index=["serif", "sans-serif", "monospace"].index(base["font_family"]) if base["font_family"] in {"serif", "sans-serif", "monospace"} else 0)
    cmap = c3.text_input("Matplotlib colormap", value=str(base["cmap"]), help="Examples: viridis, plasma, cividis, magma.")

    s1, s2, s3, s4, s5 = st.columns(5)
    title_size = s1.number_input("Title", min_value=6.0, max_value=40.0, value=float(base["font_size_title"]), step=0.5)
    label_size = s2.number_input("Axis label", min_value=6.0, max_value=40.0, value=float(base["font_size_label"]), step=0.5)
    tick_size = s3.number_input("Tick-label font size", min_value=5.0, max_value=32.0, value=float(base["font_size_tick"]), step=0.5)
    legend_size = s4.number_input("Legend", min_value=5.0, max_value=32.0, value=float(base["font_size_legend"]), step=0.5)
    legend_title_size = s5.number_input("Legend title", min_value=5.0, max_value=32.0, value=float(base["font_size_legend_title"]), step=0.5)

    st.subheader("Geometry and lines")
    g1, g2, g3, g4 = st.columns(4)
    figure_single_w = g1.number_input("Single width", min_value=2.0, max_value=20.0, value=float(base["figure_single"][0]), step=0.1)
    figure_single_h = g2.number_input("Single height", min_value=2.0, max_value=20.0, value=float(base["figure_single"][1]), step=0.1)
    figure_wide_w = g3.number_input("Wide width", min_value=2.0, max_value=24.0, value=float(base["figure_wide"][0]), step=0.1)
    figure_wide_h = g4.number_input("Wide height", min_value=2.0, max_value=20.0, value=float(base["figure_wide"][1]), step=0.1)

    l1, l2, l3, l4 = st.columns(4)
    line_width = l1.number_input("Main line", min_value=0.2, max_value=8.0, value=float(base["line_width_main"]), step=0.1)
    secondary_line = l2.number_input("Secondary line", min_value=0.2, max_value=8.0, value=float(base["line_width_secondary"]), step=0.1)
    marker_size = l3.number_input("Marker size", min_value=0.5, max_value=20.0, value=float(base["marker_size"]), step=0.2)
    cap_size = l4.number_input("Error cap length", min_value=0.0, max_value=12.0, value=float(base["cap_size"]), step=0.2)

    st.subheader("Axes, ticks and uncertainty bands")
    t1, t2, t3, t4, t5 = st.columns(5)
    axes_line_width = t1.number_input("Axis spine width", min_value=0.2, max_value=5.0, value=float(base["axes_line_width"]), step=0.1)
    tick_major_width = t2.number_input("Major tick width", min_value=0.2, max_value=5.0, value=float(base["tick_major_width"]), step=0.1)
    tick_minor_width = t3.number_input("Minor tick width", min_value=0.1, max_value=5.0, value=float(base["tick_minor_width"]), step=0.1)
    tick_major_length = t4.number_input("Major tick length", min_value=0.5, max_value=15.0, value=float(base["tick_major_length"]), step=0.5)
    tick_minor_length = t5.number_input("Minor tick length", min_value=0.5, max_value=12.0, value=float(base["tick_minor_length"]), step=0.5)
    u1, u2, u3, u4, u5 = st.columns(5)
    tick_direction = u1.selectbox("Tick direction", ["in", "out", "inout"], index=["in", "out", "inout"].index(str(base["tick_direction"])))
    ticks_top = u2.checkbox("Top ticks", value=bool(base["ticks_top"]))
    ticks_right = u3.checkbox("Right ticks", value=bool(base["ticks_right"]))
    band_alpha = u4.slider("Band opacity", min_value=0.0, max_value=1.0, value=float(base["band_alpha"]), step=0.01)
    errorbar_line_width = u5.number_input("Error-bar width", min_value=0.2, max_value=5.0, value=float(base["errorbar_line_width"]), step=0.1)
    v1, v2, v3 = st.columns(3)
    cap_thickness = v1.number_input("Error-cap thickness", min_value=0.2, max_value=5.0, value=float(base["cap_thickness"]), step=0.1)
    grid_alpha = v2.slider("Grid opacity", min_value=0.0, max_value=1.0, value=float(base["grid_alpha"]), step=0.01)
    grid_line_width = v3.number_input("Grid line width", min_value=0.1, max_value=4.0, value=float(base["grid_line_width"]), step=0.1)

    o1, o2, o3, o4 = st.columns(4)
    grid = o1.checkbox("Grid", value=bool(base["grid"]))
    use_latex = o2.checkbox("Use LaTeX", value=bool(base["use_latex"]), help="Requires a local LaTeX installation.")
    legend_frame = o3.checkbox("Legend frame", value=bool(base["legend_frame"]))
    legend_fancybox = o4.checkbox("Rounded legend", value=bool(base["legend_fancybox"]))
    q1, q2 = st.columns(2)
    legend_alpha = q1.slider("Legend opacity", min_value=0.0, max_value=1.0, value=float(base["legend_alpha"]), step=0.01)
    legend_columns = q2.number_input("Default legend columns", min_value=1, max_value=8, value=int(base["legend_columns"]), step=1)

    edited = {
        "label": label,
        "font_family": font_family,
        "mathtext_fontset": "dejavuserif" if font_family == "serif" else "dejavusans",
        "cmap": cmap,
        "grid": grid,
        "use_latex": use_latex,
        "figure_single": [figure_single_w, figure_single_h],
        "figure_wide": [figure_wide_w, figure_wide_h],
        "font_size_title": title_size,
        "font_size_label": label_size,
        "font_size_tick": tick_size,
        "font_size_legend": legend_size,
        "font_size_legend_title": legend_title_size,
        "line_width_main": line_width,
        "line_width_secondary": secondary_line,
        "marker_size": marker_size,
        "cap_size": cap_size,
        "axes_line_width": axes_line_width,
        "tick_major_width": tick_major_width,
        "tick_minor_width": tick_minor_width,
        "tick_major_length": tick_major_length,
        "tick_minor_length": tick_minor_length,
        "tick_direction": tick_direction,
        "ticks_top": ticks_top,
        "ticks_right": ticks_right,
        "band_alpha": band_alpha,
        "errorbar_line_width": errorbar_line_width,
        "cap_thickness": cap_thickness,
        "grid_alpha": grid_alpha,
        "grid_line_width": grid_line_width,
        "legend_frame": legend_frame,
        "legend_fancybox": legend_fancybox,
        "legend_alpha": legend_alpha,
        "legend_columns": int(legend_columns),
    }

    st.subheader("Preview")
    try:
        figure = style_preview(edited)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    except Exception as exc:
        st.error(f"The style preview failed: {exc}")
        return

    st.subheader("Save")
    save_name = st.text_input("Preset file name", value=selected)
    b1, b2 = st.columns(2)
    if b1.button("Save preset", type="primary", width="stretch"):
        try:
            path = save_style(PROJECT_ROOT, save_name, edited)
            st.success(f"Saved `{path.relative_to(PROJECT_ROOT)}`")
        except Exception as exc:
            st.error(str(exc))
    if b2.button("Save and set active", width="stretch"):
        try:
            path = save_style(PROJECT_ROOT, save_name, edited)
            set_active_style(PROJECT_ROOT, path.stem)
            st.success(f"Active style changed to `{path.stem}`. Regenerate products to apply it everywhere.")
        except Exception as exc:
            st.error(str(exc))
