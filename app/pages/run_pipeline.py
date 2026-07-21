from __future__ import annotations

import subprocess

import streamlit as st

from app.context import PROJECT_ROOT
from scintillation.gui.pipeline import (
    PipelineSelection,
    build_plan,
    dependency_status,
    run_plan,
)
from scintillation.gui.style_config import active_style_name, available_styles


_STAGE_KEYS = {
    "analysis": "pipe_analysis",
    "fits": "pipe_fits",
    "primary": "pipe_primary",
    "secondary": "pipe_secondary",
    "spectra": "pipe_spectra",
    "tables": "pipe_tables",
    "diagnostics": "pipe_diagnostics",
}


def _preset(**values: bool) -> None:
    for stage, key in _STAGE_KEYS.items():
        st.session_state[key] = bool(values.get(stage, False))


def page() -> None:
    st.title("Run pipeline")
    st.caption("Run analysis, fits and products through the same two shell runners used from the terminal.")

    p1, p2, p3, p4 = st.columns(4)
    p1.button("Run all preset", width="stretch", on_click=_preset,
              kwargs={stage: True for stage in _STAGE_KEYS if stage != "diagnostics"})
    p2.button("Products preset", width="stretch", on_click=_preset,
              kwargs={"primary": True, "secondary": True, "spectra": True, "tables": True})
    p3.button("Fits only preset", width="stretch", on_click=_preset, kwargs={"fits": True})
    p4.button("Clear", width="stretch", on_click=_preset, kwargs={})

    st.subheader("1 · Select stages")
    c1, c2, c3 = st.columns(3)
    analysis = c1.checkbox("Analysis", value=True, key=_STAGE_KEYS["analysis"], help="Experimental, Degrad, Garfield and spectral inputs.")
    fits = c1.checkbox("Fits", value=True, key=_STAGE_KEYS["fits"], help="Primary and joint fits with compressed toy caches.")
    primary = c2.checkbox("Primary predictions", value=True, key=_STAGE_KEYS["primary"])
    secondary = c2.checkbox("Secondary predictions", value=True, key=_STAGE_KEYS["secondary"])
    spectra = c3.checkbox("Spectra", value=True, key=_STAGE_KEYS["spectra"])
    tables = c3.checkbox("LaTeX tables", value=True, key=_STAGE_KEYS["tables"])

    with st.expander("Advanced execution options", expanded=False):
        a1, a2, a3 = st.columns(3)
        toys = a1.number_input("Stat and syst toys per fit", min_value=1, max_value=10000, value=100, step=10)
        style_names = list(available_styles(PROJECT_ROOT)) or ["tfm"]
        active = active_style_name(PROJECT_ROOT)
        style_index = style_names.index(active) if active in style_names else 0
        style = a2.selectbox("Plot style preset", style_names, index=style_index)
        diagnostics = a3.checkbox("Optional diagnostics", value=False, key=_STAGE_KEYS["diagnostics"])
        b1, b2, b3, b4 = st.columns(4)
        recompute_bands = b1.checkbox("Recompute cached bands", value=False)
        recompute_tables = b2.checkbox("Recompute cached prediction tables", value=False)
        export_scan_data = b3.checkbox("Export numerical scan CSV", value=False)
        archive_outputs = b4.checkbox("Archive existing outputs first", value=False)

    selection = PipelineSelection(
        analysis=analysis,
        fits=fits,
        primary=primary,
        secondary=secondary,
        spectra=spectra,
        tables=tables,
        diagnostics=diagnostics,
        toys=int(toys),
        recompute_bands=recompute_bands,
        recompute_tables=recompute_tables,
        export_scan_data=export_scan_data,
        archive_outputs=archive_outputs,
        style_preset=style,
    )

    st.subheader("2 · Plan and dependencies")
    try:
        plan = build_plan(selection, PROJECT_ROOT)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.code(plan.display(), language="bash", wrap_lines=True)
    checks = dependency_status(PROJECT_ROOT, selection)
    if checks:
        for label, ready, detail in checks:
            icon = "✅" if ready else "❌"
            st.markdown(f"{icon} **{label}** — `{detail}`")
    blocked = any(not ready for _, ready, _ in checks)
    if blocked:
        st.error("At least one required input is missing. Select the preceding stage or generate the missing cache first.")

    st.subheader("3 · Execute")
    st.caption("The interface streams the same stdout that you would see in the terminal. Closing the browser does not make the command reproducible; the command above does.")
    if st.button("Run selected stages", type="primary", disabled=blocked, width="stretch"):
        lines: list[str] = []
        log = st.empty()
        status = st.status("Pipeline running…", expanded=True)
        try:
            for line in run_plan(PROJECT_ROOT, plan):
                lines.append(line)
                log.code("\n".join(lines[-180:]), language="text")
            status.update(label="Pipeline completed", state="complete", expanded=False)
            st.success("Selected stages completed successfully.")
        except subprocess.CalledProcessError as exc:
            status.update(label=f"Pipeline failed with code {exc.returncode}", state="error", expanded=True)
            st.error("The workflow stopped. The final log lines are shown above.")
