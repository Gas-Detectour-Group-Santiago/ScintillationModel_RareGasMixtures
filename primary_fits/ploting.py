import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.transforms import Bbox
import scienceplots

plt.style.use("grid")


def darken_color(color, factor=0.65):
    """
    factor > 0  -> oscurece
    factor < 0  -> aclara
    factor = 0  -> no cambia
    """
    rgb = np.array(mcolors.to_rgb(color), dtype=float)

    if factor > 0:
        new_rgb = rgb * (1 - factor)
    elif factor < 0:
        a = -factor
        new_rgb = rgb + (1 - rgb) * a
    else:
        new_rgb = rgb

    return tuple(np.clip(new_rgb, 0, 1))


def _expand_bbox(bbox, pad):
    return Bbox.from_extents(
        bbox.x0 - pad, bbox.y0 - pad,
        bbox.x1 + pad, bbox.y1 + pad
    )


def _bbox_overlap_area(b1, b2):
    dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
    dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy


def _valid_mask_for_plot(x, y, xscale="log", yscale="log", xlim=None, ylim=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if xscale == "log":
        mask &= x > 0
    if yscale == "log":
        mask &= y > 0

    if xlim is not None:
        xmin, xmax = min(xlim), max(xlim)
        mask &= (x >= xmin) & (x <= xmax)

    if ylim is not None:
        ymin, ymax = min(ylim), max(ylim)
        mask &= (y >= ymin) & (y <= ymax)

    return mask


def _sample_curve_display_points(
    ax,
    x,
    y,
    *,
    xscale="log",
    yscale="log",
    xlim=None,
    ylim=None,
    step=8
):
    mask = _valid_mask_for_plot(
        x, y, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim
    )
    if not np.any(mask):
        return np.empty((0, 2))

    pts = np.column_stack([np.asarray(x)[mask], np.asarray(y)[mask]])
    pts_disp = ax.transData.transform(pts)

    if len(pts_disp) <= step:
        return pts_disp
    return pts_disp[::step]


def _sample_exp_display_points(
    ax,
    x,
    y,
    *,
    xscale="log",
    yscale="log",
    xlim=None,
    ylim=None
):
    mask = _valid_mask_for_plot(
        x, y, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim
    )
    if not np.any(mask):
        return np.empty((0, 2))

    pts = np.column_stack([np.asarray(x)[mask], np.asarray(y)[mask]])
    return ax.transData.transform(pts)


def _local_display_slope(ax, x, y, idx):
    n = len(x)
    i0 = max(0, idx - 3)
    i1 = min(n - 1, idx + 3)

    xx = np.asarray(x[i0:i1 + 1], dtype=float)
    yy = np.asarray(y[i0:i1 + 1], dtype=float)

    mask = np.isfinite(xx) & np.isfinite(yy)
    if np.count_nonzero(mask) < 2:
        return 1e6

    pts = ax.transData.transform(np.column_stack([xx[mask], yy[mask]]))
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]

    return abs(dy) / (abs(dx) + 1e-9)


def _candidate_anchor_indices(
    x,
    y,
    *,
    xscale="log",
    yscale="log",
    xlim=None,
    ylim=None,
    region=(0.60, 0.92),
    n=18
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = _valid_mask_for_plot(
        x, y, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim
    )
    idx_valid = np.where(mask)[0]

    if len(idx_valid) == 0:
        return np.array([], dtype=int)

    a, b = region
    a = float(np.clip(a, 0, 1))
    b = float(np.clip(b, 0, 1))

    i0 = int(round(a * (len(idx_valid) - 1)))
    i1 = int(round(b * (len(idx_valid) - 1)))
    if i1 < i0:
        i0, i1 = i1, i0

    idx_window = idx_valid[i0:i1 + 1]
    if len(idx_window) == 0:
        idx_window = idx_valid

    if len(idx_window) <= n:
        return idx_window

    pos = np.linspace(0, len(idx_window) - 1, n)
    return np.unique(idx_window[np.round(pos).astype(int)])


def _make_text_bbox(
    ax,
    renderer,
    text,
    xy,
    color,
    offset_pts,
    fontsize=10,
    fontweight="normal",
    bbox=False
):
    dx, dy = offset_pts

    ha = "left" if dx >= 0 else "right"

    if dy > 4:
        va = "bottom"
    elif dy < -4:
        va = "top"
    else:
        va = "center"

    ann = ax.annotate(
        text,
        xy=xy,
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=fontsize,
        fontweight=fontweight,
        ha=ha,
        va=va,
        bbox=(dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75) if bbox else None),
    )

    bb = ann.get_window_extent(renderer=renderer)
    ann.remove()
    return bb, ha, va


def _count_points_in_bbox(bbox, pts, pad=4):
    if pts.size == 0:
        return 0

    bb = _expand_bbox(bbox, pad)
    inside = (
        (pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
        (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)
    )
    return int(np.count_nonzero(inside))


def _curve_penalty_from_bbox(bbox, curve_pts, pad=3):
    if curve_pts.size == 0:
        return 0

    bb = _expand_bbox(bbox, pad)
    inside = (
        (curve_pts[:, 0] >= bb.x0) & (curve_pts[:, 0] <= bb.x1) &
        (curve_pts[:, 1] >= bb.y0) & (curve_pts[:, 1] <= bb.y1)
    )
    return int(np.count_nonzero(inside))


def _place_curve_labels_optimally(
    ax,
    curve_infos,
    exp_point_sets,
    *,
    xscale="log",
    yscale="log",
    xlim=None,
    ylim=None,
    fontsize=10,
    fontweight="normal",
    bbox=False,
):
    """
    curve_infos: lista de dicts con claves
        - x
        - y
        - label
        - color
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axes_bbox = ax.get_window_extent(renderer=renderer)

    exp_pts_all = []
    for pts in exp_point_sets:
        if pts.size:
            exp_pts_all.append(pts)
    exp_pts_all = np.vstack(exp_pts_all) if len(exp_pts_all) else np.empty((0, 2))

    sampled_curves = []
    for info in curve_infos:
        pts = _sample_curve_display_points(
            ax,
            info["x"],
            info["y"],
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            step=7
        )
        sampled_curves.append(pts)

    def sort_key(info):
        x = np.asarray(info["x"], dtype=float)
        y = np.asarray(info["y"], dtype=float)
        idxs = _candidate_anchor_indices(
            x, y,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            region=(0.75, 0.90),
            n=6
        )
        if len(idxs) == 0:
            return -np.inf
        return np.nanmedian(y[idxs])

    order = np.argsort([sort_key(info) for info in curve_infos])[::-1]

    placed_bboxes = []
    placed_annotations = []

    offsets = [
        (8, 0), (10, 8), (10, -8),
        (14, 12), (14, -12),
        (18, 0), (20, 10), (20, -10),
        (-8, 8), (-8, -8), (-14, 0),
        (26, 0), (28, 12), (28, -12),
    ]

    for k in order:
        info = curve_infos[k]
        x = np.asarray(info["x"], dtype=float)
        y = np.asarray(info["y"], dtype=float)
        label = info["label"]
        color = info["color"]

        idx_candidates = _candidate_anchor_indices(
            x, y,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            region=(0.62, 0.93),
            n=18
        )

        if len(idx_candidates) == 0:
            continue

        best = None
        best_score = np.inf

        for idx in idx_candidates:
            xy = (x[idx], y[idx])

            if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                continue

            slope_pen = _local_display_slope(ax, x, y, idx)

            for offset in offsets:
                bb, ha, va = _make_text_bbox(
                    ax,
                    renderer,
                    label,
                    xy,
                    color=darken_color(color, 0.20),
                    offset_pts=offset,
                    fontsize=fontsize,
                    fontweight=fontweight,
                    bbox=bbox
                )

                out_pen = 0.0
                if (
                    bb.x0 < axes_bbox.x0 or bb.x1 > axes_bbox.x1 or
                    bb.y0 < axes_bbox.y0 or bb.y1 > axes_bbox.y1
                ):
                    out_pen += 1e9

                overlap_pen = 0.0
                for bb_prev in placed_bboxes:
                    overlap_pen += 2000.0 * _bbox_overlap_area(bb, _expand_bbox(bb_prev, 2))

                point_pen = 12000.0 * _count_points_in_bbox(bb, exp_pts_all, pad=4)

                curve_pen = 0.0
                for j, pts_curve in enumerate(sampled_curves):
                    if pts_curve.size == 0:
                        continue

                    w = 350.0 if j == k else 900.0
                    curve_pen += w * _curve_penalty_from_bbox(bb, pts_curve, pad=3)

                slope_term = 120.0 * slope_pen

                anchor_disp = ax.transData.transform([xy])[0]
                right_reward = -0.08 * anchor_disp[0]

                dx, dy = offset
                offset_pen = 0.8 * np.hypot(dx, dy)

                score = (
                    out_pen +
                    overlap_pen +
                    point_pen +
                    curve_pen +
                    slope_term +
                    offset_pen +
                    right_reward
                )

                if score < best_score:
                    best_score = score
                    best = (xy, offset, bb, ha, va)

        if best is None:
            continue

        xy, offset, bb, ha, va = best

        ann = ax.annotate(
            label,
            xy=xy,
            xytext=offset,
            textcoords="offset points",
            color=darken_color(color, 0.20),
            fontsize=fontsize,
            fontweight=fontweight,
            ha=ha,
            va=va,
            bbox=(dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75) if bbox else None),
        )

        placed_bboxes.append(bb)
        placed_annotations.append(ann)

    return placed_annotations


def plot_fit_vs_experiment_by_pressure(
    df_exp,
    theory_func,
    fit_params,
    degrad_data,
    concentration_grid,
    *,
    x_col="fCF4",
    pressure_cols=None,
    pressures=None,
    pressure_regex=r"^\s*([0-9]+(?:\.[0-9]+)?)\s*bar\s*$",
    err_patterns=None,
    x_plot_factor=100.0,
    min_positive_x=None,
    cmap="viridis",
    darken_factor=0.65,
    figsize=(6, 4),
    title=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    xscale="log",
    yscale="log",
    line_label_fmt=None,
    exp_label_fmt="{p:g} bar exp",
    legend=True,
    legend_kwargs=None,
    output=None,
    show=True,
    ax=None,
    activate_components=False,
    label_mode="legend",
    annotate_fmt="{p:g} bar",
    annotate_fontsize=10,
    annotate_fontweight="normal",
    annotate_bbox=False,
):
    """
    label_mode:
        - "legend"   -> usa leyenda normal
        - "annotate" -> escribe etiquetas cerca de la curva y no usa leyenda
    """
    norm = fit_params[0]
    if err_patterns is None:
        err_patterns = [
            "Err {col}",
            "Err_{col}",
            "{col} Err",
            "{col}_Err",
        ]

    if line_label_fmt is None:
        line_label_fmt = ["{p:g} bar fit"]

    if label_mode not in {"legend", "annotate"}:
        raise ValueError("label_mode debe ser 'legend' o 'annotate'.")

    concentration_grid = np.asarray(concentration_grid, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.grid(False) #, which="major", alpha=0.3)
    ax.grid(False) #, which="minor", alpha=0.08)

    x_exp = df_exp[x_col].to_numpy(dtype=float).copy()
    if xscale == "log" and min_positive_x is not None:
        x_exp[x_exp <= 0] = min_positive_x

    x_exp_plot = x_exp
    x_grid_plot = concentration_grid * x_plot_factor

    if pressure_cols is None:
        pressure_cols = []
        regex = re.compile(pressure_regex)

        for col in df_exp.columns:
            col_str = str(col)

            if col_str == x_col:
                continue

            m = regex.match(col_str)
            if m:
                pressure_cols.append((float(m.group(1)), col_str))

        pressure_cols.sort(key=lambda t: t[0])

    if not pressure_cols:
        raise ValueError("No se encontraron columnas de presión válidas.")

    if pressures is not None:
        pressures_set = {float(p) for p in pressures}
        pressure_cols = [(p, col) for p, col in pressure_cols if p in pressures_set]

        if not pressure_cols:
            raise ValueError(
                f"Ninguna de las presiones pedidas {pressures} está en los datos."
            )

    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(0.15, 0.85, len(pressure_cols)))

    linestyles = [
        "--",
        "-.",
        ":",
        (0, (1, 1)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (5, 2, 1, 2)),
        (0, (10, 2)),
        (0, (3, 5, 1, 5)),
    ]

    annotate_curve_infos = []
    exp_point_sets = []

    for (p, col), color in zip(pressure_cols, colors):
        point_color = darken_color(color, factor=darken_factor)

        y_fit = np.asarray(
            theory_func(
                fit_params,
                degrad_data,
                concentration_grid,
                p,
                activate_components=activate_components
            ),
            dtype=float
        )

        if activate_components:
            for i, y in enumerate(y_fit):
                curve_color = darken_color(colors[i % len(colors)], 0.3)

                use_label = None
                if label_mode == "legend" and i < len(line_label_fmt):
                    use_label = line_label_fmt[i].format(p=p)

                if i == 0:
                    ax.plot(
                        x_grid_plot,
                        y,
                        color=curve_color,
                        lw=2,
                        label=use_label
                    )

                    if label_mode == "annotate":
                        annotate_curve_infos.append({
                            "x": x_grid_plot.copy(),
                            "y": np.asarray(y, dtype=float).copy(),
                            "label": annotate_fmt.format(p=p),
                            "color": curve_color,
                        })

                elif i < 10:
                    ax.plot(
                        x_grid_plot,
                        y,
                        color=curve_color,
                        linestyle=linestyles[i % len(linestyles)],
                        lw=2,
                        label=use_label
                    )
        else:
            curve_color = darken_color(color, 0.3)
            use_label = line_label_fmt[0].format(p=p) if label_mode == "legend" else None

            ax.plot(
                x_grid_plot,
                y_fit,
                color=curve_color,
                lw=2,
                label=use_label
            )

            if label_mode == "annotate":
                annotate_curve_infos.append({
                    "x": x_grid_plot.copy(),
                    "y": np.asarray(y_fit, dtype=float).copy(),
                    "label": annotate_fmt.format(p=p),
                    "color": curve_color,
                })

        err_col = None
        for pattern in err_patterns:
            candidate = pattern.format(col=col)
            if candidate in df_exp.columns:
                err_col = candidate
                break

        y_exp = df_exp[col].to_numpy(dtype=float)
        yerr = df_exp[err_col].to_numpy(dtype=float) if err_col is not None else None

        exp_label = exp_label_fmt.format(p=p) if label_mode == "legend" else None

        ax.errorbar(
            x_exp_plot,
            y_exp,
            yerr=yerr,
            fmt="o",
            ms=4,
            color=point_color,
            ecolor=point_color,
            elinewidth=1,
            capsize=2,
            label=exp_label
        )

        exp_point_sets.append(
            _sample_exp_display_points(
                ax,
                x_exp_plot,
                y_exp,
                xscale=xscale,
                yscale=yscale,
                xlim=xlim,
                ylim=ylim
            )
        )


    secax = ax.secondary_yaxis(
        'right',
        functions=(lambda y: y*1000/norm, lambda y: y*1000/norm)
    )
    secax.set_ylabel("ph / MeV")

    ax.tick_params(axis='y', which='both', right=False, labelright=False)
    secax.tick_params(axis='y', which='both', left=False, labelleft=False)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if label_mode == "annotate" and len(annotate_curve_infos) > 0:
        _place_curve_labels_optimally(
            ax,
            annotate_curve_infos,
            exp_point_sets,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            fontsize=annotate_fontsize,
            fontweight=annotate_fontweight,
            bbox=annotate_bbox,
        )

    if legend and label_mode == "legend":
        if legend_kwargs is None:
            legend_kwargs = {}
        ax.legend(**legend_kwargs)

    if output is not None:
        fig.savefig(output, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, pressure_cols