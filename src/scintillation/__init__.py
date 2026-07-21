"""ScintillationModel public package.

The package provides stable registries, runtime adapters and common result
contracts. The validated legacy physics implementation is executed through a
compatibility workspace while models are migrated module by module.
"""

from .core.paths import ProjectPaths, find_project_root
from .core.registry import ProjectRegistry
from .core.outputs import OutputManager

__all__ = ["ProjectPaths", "ProjectRegistry", "OutputManager", "find_project_root"]
