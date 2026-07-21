from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import find_project_root


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MixtureSpec:
    mixture_id: str
    label: str
    base_gas: str
    additive: str | None
    primary_fit: str | None
    ir_fit: str | None
    default_xray_energy_keV: float
    default_normalization: str
    second_continuum: bool
    status: str
    notes: str = ""

    @property
    def active(self) -> bool:
        return self.status.lower() == "active"


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    label: str
    model_id: str
    parameter_family: str
    wavelength_min_nm: float | None
    wavelength_max_nm: float | None
    primary_enabled: bool
    secondary_enabled: bool
    default_normalization: str
    status: str


@dataclass(frozen=True)
class NormalizationSpec:
    normalization_id: str
    mode: str
    reference_fit: str | None
    output_scale: float
    output_unit: str
    propagate_nnorm: bool
    description: str


@dataclass(frozen=True)
class FitSpec:
    fit_name: str
    mixture_id: str
    model_id: str
    channel_group: str
    parameter_file: Path
    enabled: bool
    notes: str


@dataclass(frozen=True)
class SecondaryParameterSetSpec:
    set_id: str
    mixture_id: str
    channel_id: str
    model_id: str
    base_fit_name: str | None
    kinetic_parameter_family: str | None
    normalization_recipe: str
    ocw_recipe: str | None
    parameter_transform: str
    uncertainty_sources: tuple[str, ...]
    status: str
    notes: str

    @property
    def active(self) -> bool:
        return self.status.lower() == "active"


class ProjectRegistry:
    """Single source of truth for mixtures, channels, fits and normalizations."""

    def __init__(self, root: Path, mixtures: dict[str, MixtureSpec], channels: dict[str, ChannelSpec],
                 normalizations: dict[str, NormalizationSpec], fits: dict[str, FitSpec],
                 secondary_parameter_sets: dict[str, SecondaryParameterSetSpec]):
        self.root = Path(root)
        self.mixtures = mixtures
        self.channels = channels
        self.normalizations = normalizations
        self.fits = fits
        self.secondary_parameter_sets = secondary_parameter_sets

    @classmethod
    def load(cls, root: str | Path) -> "ProjectRegistry":
        root = find_project_root(root)
        mixtures_df = pd.read_csv(root / "config" / "mixtures.csv")
        channels_df = pd.read_csv(root / "config" / "channels.csv")
        norms_df = pd.read_csv(root / "config" / "normalizations.csv")
        fits_df = pd.read_csv(root / "config" / "fits.csv")
        secondary_sets_path = root / "config" / "secondary_parameter_sets.csv"
        secondary_sets_df = pd.read_csv(secondary_sets_path) if secondary_sets_path.exists() else pd.DataFrame()

        mixtures = {
            str(r.mixture_id): MixtureSpec(
                mixture_id=str(r.mixture_id), label=str(r.label), base_gas=str(r.base_gas),
                additive=_optional_text(r.additive), primary_fit=_optional_text(r.primary_fit),
                ir_fit=_optional_text(r.ir_fit), default_xray_energy_keV=float(r.default_xray_energy_keV),
                default_normalization=str(r.default_normalization), second_continuum=_as_bool(r.second_continuum),
                status=str(r.status), notes=str(getattr(r, "notes", "") or ""),
            ) for r in mixtures_df.itertuples(index=False)
        }
        channels = {
            str(r.channel_id): ChannelSpec(
                channel_id=str(r.channel_id), label=str(r.label), model_id=str(r.model_id),
                parameter_family=str(r.parameter_family),
                wavelength_min_nm=None if pd.isna(r.wavelength_min_nm) else float(r.wavelength_min_nm),
                wavelength_max_nm=None if pd.isna(r.wavelength_max_nm) else float(r.wavelength_max_nm),
                primary_enabled=_as_bool(r.primary_enabled), secondary_enabled=_as_bool(r.secondary_enabled),
                default_normalization=str(r.default_normalization), status=str(r.status),
            ) for r in channels_df.itertuples(index=False)
        }
        normalizations = {
            str(r.normalization_id): NormalizationSpec(
                normalization_id=str(r.normalization_id), mode=str(r.mode),
                reference_fit=_optional_text(r.reference_fit), output_scale=float(r.output_scale),
                output_unit=str(r.output_unit), propagate_nnorm=_as_bool(r.propagate_nnorm),
                description=str(r.description),
            ) for r in norms_df.itertuples(index=False)
        }
        fits = {
            str(r.fit_name): FitSpec(
                fit_name=str(r.fit_name), mixture_id=str(r.mixture_id), model_id=str(r.model_id),
                channel_group=str(r.channel_group), parameter_file=root / str(r.parameter_file),
                enabled=_as_bool(r.enabled), notes=str(getattr(r, "notes", "") or ""),
            ) for r in fits_df.itertuples(index=False)
        }
        secondary_parameter_sets = {}
        if not secondary_sets_df.empty:
            secondary_parameter_sets = {
                str(r.set_id): SecondaryParameterSetSpec(
                    set_id=str(r.set_id), mixture_id=str(r.mixture_id), channel_id=str(r.channel_id),
                    model_id=str(r.model_id), base_fit_name=_optional_text(r.base_fit_name),
                    kinetic_parameter_family=_optional_text(r.kinetic_parameter_family),
                    normalization_recipe=str(r.normalization_recipe), ocw_recipe=_optional_text(r.ocw_recipe),
                    parameter_transform=str(r.parameter_transform),
                    uncertainty_sources=tuple(
                        item.strip() for item in str(r.uncertainty_sources).split(";") if item.strip()
                    ),
                    status=str(r.status), notes=str(getattr(r, "notes", "") or ""),
                ) for r in secondary_sets_df.itertuples(index=False)
            }
        registry = cls(root, mixtures, channels, normalizations, fits, secondary_parameter_sets)
        registry.validate()
        return registry

    def validate(self) -> None:
        errors: list[str] = []
        for mixture in self.mixtures.values():
            if mixture.default_normalization not in self.normalizations:
                errors.append(f"{mixture.mixture_id}: unknown normalization {mixture.default_normalization}")
            for fit_name in (mixture.primary_fit, mixture.ir_fit):
                if fit_name and fit_name not in self.fits:
                    errors.append(f"{mixture.mixture_id}: unknown fit {fit_name}")
        for channel in self.channels.values():
            if channel.default_normalization not in self.normalizations:
                errors.append(f"{channel.channel_id}: unknown normalization {channel.default_normalization}")
        for fit in self.fits.values():
            if fit.mixture_id not in self.mixtures:
                errors.append(f"{fit.fit_name}: unknown mixture {fit.mixture_id}")
        for parameter_set in self.secondary_parameter_sets.values():
            if parameter_set.mixture_id not in self.mixtures:
                errors.append(f"{parameter_set.set_id}: unknown mixture {parameter_set.mixture_id}")
            if parameter_set.channel_id not in self.channels:
                errors.append(f"{parameter_set.set_id}: unknown channel {parameter_set.channel_id}")
            if parameter_set.base_fit_name and parameter_set.base_fit_name not in self.fits:
                errors.append(f"{parameter_set.set_id}: unknown base fit {parameter_set.base_fit_name}")
        if errors:
            raise ValueError("Invalid project registry:\n- " + "\n- ".join(errors))

    def mixture(self, mixture_id: str) -> MixtureSpec:
        try:
            return self.mixtures[mixture_id]
        except KeyError as exc:
            raise KeyError(f"Unknown mixture {mixture_id!r}; available: {sorted(self.mixtures)}") from exc

    def channel(self, channel_id: str) -> ChannelSpec:
        try:
            return self.channels[channel_id]
        except KeyError as exc:
            raise KeyError(f"Unknown channel {channel_id!r}; available: {sorted(self.channels)}") from exc

    def normalization(self, normalization_id: str) -> NormalizationSpec:
        try:
            return self.normalizations[normalization_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown normalization {normalization_id!r}; available: {sorted(self.normalizations)}"
            ) from exc

    def enabled_mixtures(self, *, include_planned: bool = False) -> Iterable[MixtureSpec]:
        return tuple(
            item for item in self.mixtures.values()
            if include_planned or item.active
        )

    def secondary_parameter_set(self, set_id: str) -> SecondaryParameterSetSpec:
        try:
            return self.secondary_parameter_sets[set_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown secondary parameter set {set_id!r}; available: {sorted(self.secondary_parameter_sets)}"
            ) from exc
