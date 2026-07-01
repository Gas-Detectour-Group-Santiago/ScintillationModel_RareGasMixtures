"""Configurable experimental spectrum integral comparisons."""

from .aux.integrators import IntegralConfig, IntegralResult, integrate_spectrum
from .aux.spectra_io import SpectrumData, SpectrumProvider, SpectrumSelector
from .aux.ratios import IntegralDefinition, RatioDefinition, RatioScanner

__all__ = [
    "IntegralConfig",
    "IntegralDefinition",
    "IntegralResult",
    "RatioDefinition",
    "RatioScanner",
    "SpectrumData",
    "SpectrumProvider",
    "SpectrumSelector",
    "integrate_spectrum",
]
