from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..core.paths import ProjectPaths


def build_output_inventory(project_root: str | Path) -> pd.DataFrame:
    paths=ProjectPaths.from_root(project_root)
    rows=[]
    for path in paths.current.rglob("*"):
        if path.is_file():
            rows.append({"path":str(path.relative_to(paths.root)),"suffix":path.suffix.lower(),"size_bytes":path.stat().st_size})
    frame=pd.DataFrame(rows)
    out=paths.current/"report/output_inventory.csv"; out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(out,index=False)
    return frame
