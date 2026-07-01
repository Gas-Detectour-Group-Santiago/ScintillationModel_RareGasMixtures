"""Auxiliary classes and functions for integral comparisons."""

from .integrators import IntegralConfig, IntegralResult, integrate_spectrum
from .ratios import IntegralDefinition, RatioDefinition, RatioScanner, ScanConfig
from .spectra_io import SpectrumData, SpectrumProvider, SpectrumSelector

__all__ = [
    "IntegralConfig",
    "IntegralDefinition",
    "IntegralResult",
    "RatioDefinition",
    "RatioScanner",
    "ScanConfig",
    "SpectrumData",
    "SpectrumProvider",
    "SpectrumSelector",
    "integrate_spectrum",
]
