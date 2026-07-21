from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from app.context import PROJECT_ROOT

FAMILIES = {
    "Fits": "fits",
    "Primary predictions": "primary",
    "Secondary predictions": "secondary",
    "Spectra": "spectra",
}


def _path(family: str) -> Path:
    return PROJECT_ROOT / "config" / "plots" / f"{family}.csv"


def _validate(frame: pd.DataFrame, family: str) -> list[str]:
    errors: list[str] = []
    required = {"plot_id", "enabled", "output"}
    missing = required - set(frame.columns)
    if missing:
        errors.append(f"Missing columns: {', '.join(sorted(missing))}")
    if "plot_id" in frame and frame["plot_id"].astype(str).str.strip().eq("").any():
        errors.append("plot_id cannot be empty")
    if family != "spectra" and "series_id" in frame:
        duplicate = frame.duplicated(["plot_id", "series_id"], keep=False)
        if duplicate.any():
            errors.append("plot_id + series_id must be unique for every curve/layer")
    if "enabled" in frame:
        allowed = {"true", "false", "1", "0", "yes", "no", "active", "disabled"}
        invalid = ~frame["enabled"].astype(str).str.lower().isin(allowed)
        if invalid.any():
            errors.append("enabled accepts true/false")
    return errors


def page() -> None:
    st.title("Figure recipes")
    st.caption(
        "These four CSV files are the canonical production-figure registry. "
        "Rows with the same plot_id belong to the same PDF; each row is one curve, dataset or spectral recipe."
    )
    tabs = st.tabs(list(FAMILIES))
    for tab, (label, family) in zip(tabs, FAMILIES.items(), strict=True):
        with tab:
            path = _path(family)
            if not path.exists():
                st.error(f"Missing `{path.relative_to(PROJECT_ROOT)}`")
                continue
            frame = pd.read_csv(path, keep_default_na=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", len(frame))
            c2.metric("PDF recipes", frame["plot_id"].nunique() if "plot_id" in frame else 0)
            c3.metric("Enabled rows", int(frame["enabled"].astype(str).str.lower().isin({"true", "1", "yes", "active"}).sum()) if "enabled" in frame else len(frame))
            selected_plot = st.selectbox(
                "Inspect one plot",
                ["All"] + sorted(frame["plot_id"].astype(str).unique().tolist()),
                key=f"recipe_filter_{family}",
            )
            shown = frame if selected_plot == "All" else frame.loc[frame["plot_id"].astype(str) == selected_plot]
            st.dataframe(shown, hide_index=True, width="stretch", height=260)
            st.divider()
            st.markdown("#### Edit CSV")
            edited = st.data_editor(
                frame,
                hide_index=True,
                num_rows="dynamic",
                width="stretch",
                height=520,
                key=f"recipe_editor_{family}",
            )
            errors = _validate(edited, family)
            if errors:
                for error in errors:
                    st.error(error)
            if st.button(f"Save {label}", type="primary", disabled=bool(errors), key=f"save_{family}"):
                edited.to_csv(path, index=False)
                st.success(f"Saved `{path.relative_to(PROJECT_ROOT)}`. Run the corresponding stage to regenerate PDFs.")
            with st.expander("How this family works"):
                if family == "fits":
                    st.write("One row per fit diagnostic. Experimental points always use statistical error bars; enabled controls each PDF.")
                elif family == "primary":
                    st.write("One row per model component/curve. datasets references config/experimental_datasets.csv; bands and normalization share the common registries.")
                elif family == "secondary":
                    st.write("One row per model, combined component, experimental series, metadata curve or transport scan. selection_id points to config/secondary_selections.csv.")
                else:
                    st.write("One row per spectral figure. Concentrations define panels; pressures define curves. Mosaics, inset and broken-x are configured directly here.")
