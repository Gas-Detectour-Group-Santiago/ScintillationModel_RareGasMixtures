from __future__ import annotations

import sys
from pathlib import Path

# Makes `import spectra_generator` work when Python/Jupyter is started
# from inside the spectra_generator directory itself.
_here = Path(__file__).resolve().parent
_root = _here.parent
if (_root / "data").exists() and (_root / "models").exists():
    root_str = str(_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
