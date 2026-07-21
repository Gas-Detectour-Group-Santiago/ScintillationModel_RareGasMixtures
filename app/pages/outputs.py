from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st

from app.context import PROJECT_ROOT


def _files(kind: str, family: str) -> list[Path]:
    root = PROJECT_ROOT / "outputs" / kind
    if family != "All":
        root = root / family.lower()
    pattern = "*.pdf" if kind == "figures" else "*.tex"
    return sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []


def page() -> None:
    st.title("Outputs")
    st.caption("Browse the official PDFs and LaTeX tables. Numerical caches remain in data/cache and are not duplicated here.")
    c1, c2 = st.columns(2)
    kind = c1.radio("Artifact", ["figures", "tables"], horizontal=True)
    family = c2.selectbox("Family", ["All", "fits", "primary", "secondary", "spectra", "diagnostics", "reference"])
    files = _files(kind, family)
    st.metric("Matching outputs", len(files))
    if not files:
        st.info("No matching outputs have been generated yet.")
        return
    labels = [str(path.relative_to(PROJECT_ROOT)) for path in files]
    selected = files[labels.index(st.selectbox("File", labels))]
    m1, m2, m3 = st.columns(3)
    stat = selected.stat()
    m1.metric("Size", f"{stat.st_size / 1024:.1f} KiB")
    m2.metric("Folder", selected.parent.name)
    m3.metric("Extension", selected.suffix)
    st.code(str(selected.relative_to(PROJECT_ROOT)))
    data = selected.read_bytes()
    st.download_button("Download selected output", data=data, file_name=selected.name, mime="application/pdf" if selected.suffix == ".pdf" else "text/x-tex")
    if selected.suffix == ".tex":
        st.code(data.decode("utf-8", errors="replace"), language="latex")
    else:
        encoded = base64.b64encode(data).decode("ascii")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="760" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )
