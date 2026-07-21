"""Graphical control layer over the canonical CSV registries and shell runners."""

from .catalog import DatasetInfo, discover_primary_datasets, discover_secondary_datasets

__all__ = [
    "DatasetInfo",
    "discover_primary_datasets",
    "discover_secondary_datasets",
]
