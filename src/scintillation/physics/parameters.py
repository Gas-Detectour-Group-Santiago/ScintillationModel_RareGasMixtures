from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from ..core.outputs import OutputManager
from ..core.registry import ProjectRegistry


REGISTRY_COLUMNS = [
    "parameter_family",
    "model_id",
    "scope",
    "mixture_id",
    "additive",
    "fit_name",
    "channel_group",
    "name",
    "physical_parameter_id",
    "comparison_group",
    "process",
    "state",
    "collision_partner",
    "spectral_channel",
    "tex_name",
    "value",
    "fit_uncertainty",
    "stat_minus",
    "stat_plus",
    "syst_minus",
    "syst_plus",
    "total_minus",
    "total_plus",
    "unit",
    "fixed",
    "reference",
    "status",
    "source_file",
    "description",
]


def _text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def _number(value: object) -> float:
    if value is None or pd.isna(value) or value == "":
        return float("nan")
    return float(value)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _alias_rules(project: ProjectRegistry) -> pd.DataFrame:
    path = project.root / "config" / "parameter_aliases.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).sort_values("priority", ascending=False)
    return frame


def _semantic_fields(
    project: ProjectRegistry,
    *,
    name: str,
    model_id: str,
    fit_name: str,
    additive: str,
    channel_group: str,
) -> dict[str, str]:
    """Return a unique parameter identity plus optional comparison semantics.

    Fit parameters are always namespaced by the exact fit that produced them.
    A comparison group can relate analogous quantities across models, but it is
    never used to substitute, average or prefer one fit over another.
    """
    context = {
        "name": name,
        "model_id": model_id,
        "fit_name": fit_name,
        "additive": additive,
        "channel_group": channel_group,
    }
    if fit_name:
        parameter_id = f"fit.{fit_name}.{name}"
    else:
        scope_tag = additive or "common"
        parameter_id = f"model.{model_id}.{scope_tag}.{name}" if model_id else name

    semantic = {
        "physical_parameter_id": parameter_id,
        "comparison_group": "",
        "process": "",
        "state": "",
        "collision_partner": additive,
        "spectral_channel": channel_group,
    }
    for row in _alias_rules(project).itertuples(index=False):
        rule_model = _text(getattr(row, "model_id", ""))
        if rule_model and rule_model != model_id:
            continue
        match = re.match(str(row.pattern), name)
        if not match:
            continue

        def render(value: object) -> str:
            template = _text(value)
            if not template:
                return ""
            expanded = match.expand(template)
            return expanded.format(**context).strip()

        semantic.update({
            "comparison_group": render(getattr(row, "comparison_group_template", "")),
            "process": render(row.process),
            "state": render(row.state_template),
            "collision_partner": render(row.collision_partner_template),
            "spectral_channel": render(row.spectral_channel_template),
        })
        break
    return semantic


@dataclass(frozen=True)
class ParameterQuery:
    name: str
    model_id: str | None = None
    parameter_family: str | None = None
    mixture_id: str | None = None
    additive: str | None = None
    fit_name: str | None = None


class ParameterRegistry:
    """Long-format parameter database combining fits and literature kinetics.

    Resolution order is explicit rather than hidden in model code:
    exact fit -> exact mixture -> exact additive -> common model parameter.
    """

    def __init__(self, frame: pd.DataFrame):
        missing = set(REGISTRY_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Parameter registry misses columns: {sorted(missing)}")
        self.frame = frame.loc[:, REGISTRY_COLUMNS].copy()

    @classmethod
    def read(cls, path: str | Path) -> "ParameterRegistry":
        return cls(pd.read_csv(path))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)
        return path

    def query(self, query: ParameterQuery) -> pd.DataFrame:
        selected = self.frame.loc[self.frame["name"].astype(str) == query.name].copy()
        for column, value in (
            ("model_id", query.model_id),
            ("parameter_family", query.parameter_family),
            ("mixture_id", query.mixture_id),
            ("additive", query.additive),
            ("fit_name", query.fit_name),
        ):
            if value is not None:
                selected = selected.loc[selected[column].fillna("").astype(str) == str(value)]
        return selected.reset_index(drop=True)

    def resolve(self, name: str, *, model_id: str | None = None,
                parameter_family: str | None = None, mixture_id: str | None = None,
                additive: str | None = None, fit_name: str | None = None) -> pd.Series:
        """Resolve one exact parameter without crossing model or fit boundaries.

        For fitted parameters, ``fit_name`` is required whenever more than one
        active fit contains the requested name. Literature/common parameters can
        be resolved by exact model and additive scope.
        """
        candidates = self.frame.loc[self.frame["name"].astype(str) == name].copy()
        candidates = candidates.loc[candidates["status"].fillna("active") == "active"]
        candidates = candidates.loc[pd.to_numeric(candidates["value"], errors="coerce").notna()]
        for column, value in (
            ("fit_name", fit_name),
            ("model_id", model_id),
            ("parameter_family", parameter_family),
            ("mixture_id", mixture_id),
            ("additive", additive),
        ):
            if value is not None:
                candidates = candidates.loc[candidates[column].fillna("").astype(str) == str(value)]

        if candidates.empty:
            raise KeyError(
                f"No parameter {name!r} for model={model_id!r}, family={parameter_family!r}, "
                f"mixture={mixture_id!r}, additive={additive!r}, fit={fit_name!r}"
            )
        if len(candidates) != 1:
            choices = candidates[["fit_name", "model_id", "mixture_id", "additive", "physical_parameter_id"]]
            raise ValueError(
                f"Ambiguous parameter {name!r}. Select the exact fit/model parameter set; "
                f"available rows: {choices.to_dict(orient='records')}"
            )
        return candidates.iloc[0]

    def comparison_summary(self) -> pd.DataFrame:
        """Build side-by-side comparison groups without selecting a preferred fit."""
        compared = self.frame.loc[self.frame["comparison_group"].fillna("").astype(str) != ""].copy()
        if compared.empty:
            return pd.DataFrame(columns=[
                "comparison_group", "n_estimates", "fits", "models", "mixtures", "min_value", "max_value"
            ])
        return (
            compared.groupby("comparison_group", dropna=False)
            .agg(
                n_estimates=("value", "count"),
                fits=("fit_name", lambda values: ";".join(sorted({str(v) for v in values if str(v)}))),
                models=("model_id", lambda values: ";".join(sorted({str(v) for v in values if str(v)}))),
                mixtures=("mixture_id", lambda values: ";".join(sorted({str(v) for v in values if str(v)}))),
                min_value=("value", "min"),
                max_value=("value", "max"),
            )
            .reset_index()
        )

    def resolved_physical_parameters(self) -> pd.DataFrame:
        raise RuntimeError(
            "Automatic cross-fit parameter resolution is disabled. "
            "Use an exact fit_name/model parameter set; comparison_group is diagnostic only."
        )

    def mixture_bundle(self, project: ProjectRegistry, mixture_id: str) -> pd.DataFrame:
        mixture = project.mixture(mixture_id)
        keep = (
            (self.frame["mixture_id"].fillna("") == mixture_id)
            | (self.frame["additive"].fillna("") == (mixture.additive or ""))
            | (self.frame["scope"] == "common")
        )
        return self.frame.loc[keep].sort_values(
            ["parameter_family", "model_id", "fit_name", "scope", "name"],
            na_position="last",
        ).reset_index(drop=True)


def _fit_rows(project: ProjectRegistry) -> Iterable[dict[str, object]]:
    for fit in project.fits.values():
        if not fit.enabled or not fit.parameter_file.exists():
            continue
        frame = pd.read_csv(fit.parameter_file)
        for _, row in frame.iterrows():
            name = _text(row.get("name"))
            additive = project.mixture(fit.mixture_id).additive or ""
            semantic = _semantic_fields(
                project,
                name=name,
                model_id=fit.model_id,
                fit_name=fit.fit_name,
                additive=additive,
                channel_group=fit.channel_group,
            )
            yield {
                "parameter_family": fit.fit_name,
                "model_id": fit.model_id,
                "scope": "fit",
                "mixture_id": fit.mixture_id,
                "additive": additive,
                "fit_name": fit.fit_name,
                "channel_group": fit.channel_group,
                "name": name,
                **semantic,
                "tex_name": _text(row.get("tex_name")),
                "value": _number(row.get("value")),
                "fit_uncertainty": _number(row.get("fit_uncertainty")),
                "stat_minus": _number(row.get("stat_minus")),
                "stat_plus": _number(row.get("stat_plus")),
                "syst_minus": _number(row.get("syst_minus")),
                "syst_plus": _number(row.get("syst_plus")),
                "total_minus": _number(row.get("total_minus")),
                "total_plus": _number(row.get("total_plus")),
                "unit": _text(row.get("unit")),
                "fixed": _bool(row.get("fixed", False)),
                "reference": "fit",
                "status": "active",
                "source_file": str(fit.parameter_file.relative_to(project.root)),
                "description": fit.notes,
            }


def _second_continuum_rows(project: ProjectRegistry) -> Iterable[dict[str, object]]:
    path = project.root / "data" / "cache" / "fits" / "parameters" / "Ar2nd_continium.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        value = _number(row.get("value"))
        enabled = _bool(row.get("enabled", True))
        status = _text(row.get("status"), "active" if enabled and np.isfinite(value) else "planned")
        additive = _text(row.get("additive"))
        mixture_id = ""
        if additive:
            matches = [m.mixture_id for m in project.mixtures.values() if (m.additive or "").upper() == additive.upper()]
            mixture_id = matches[0] if matches else f"Ar{additive}"
        name = _text(row.get("name"))
        semantic = _semantic_fields(
            project,
            name=name,
            model_id="argon_second_continuum",
            fit_name="",
            additive=additive,
            channel_group="vuv",
        )
        yield {
            "parameter_family": "argon_second_continuum",
            "model_id": "argon_second_continuum",
            "scope": _text(row.get("scope"), "common"),
            "mixture_id": mixture_id,
            "additive": additive,
            "fit_name": "",
            "channel_group": "vuv",
            "name": name,
            **semantic,
            "tex_name": _text(row.get("tex_name")),
            "value": value,
            "fit_uncertainty": _number(row.get("fit_uncertainty")),
            "stat_minus": _number(row.get("stat_minus")),
            "stat_plus": _number(row.get("stat_plus")),
            "syst_minus": _number(row.get("syst_minus")),
            "syst_plus": _number(row.get("syst_plus")),
            "total_minus": _number(row.get("total_minus")),
            "total_plus": _number(row.get("total_plus")),
            "unit": _text(row.get("unit")),
            "fixed": _bool(row.get("fixed", True)),
            "reference": _text(row.get("reference")),
            "status": status,
            "source_file": str(path.relative_to(project.root)),
            "description": _text(row.get("description")),
        }


def _requirements_report(project: ProjectRegistry, registry: ParameterRegistry) -> pd.DataFrame:
    requirements = pd.read_csv(project.root / "config" / "model_requirements.csv")
    rows: list[dict[str, object]] = []
    for mixture in project.mixtures.values():
        if not mixture.second_continuum:
            continue
        for _, requirement in requirements.loc[requirements["model_id"] == "argon_second_continuum"].iterrows():
            scope = str(requirement["scope"])
            if scope == "additive" and not mixture.additive:
                continue
            additive = mixture.additive if scope == "additive" else None
            matches = registry.query(ParameterQuery(
                name=str(requirement["name"]), model_id="argon_second_continuum",
                mixture_id=mixture.mixture_id if scope == "additive" else None,
                additive=additive,
            ))
            valid = not matches.empty and matches["value"].apply(lambda x: np.isfinite(float(x))).any()
            rows.append({
                "mixture_id": mixture.mixture_id,
                "status": mixture.status,
                "model_id": "argon_second_continuum",
                "scope": scope,
                "additive": additive or "",
                "parameter": str(requirement["name"]),
                "unit": str(requirement["unit"]),
                "required": _bool(requirement["required"]),
                "available": bool(valid),
                "ready": bool(valid or not _bool(requirement["required"])),
            })
    return pd.DataFrame(rows)


def build_project_parameter_registry(project_root: str | Path, *, run_name: str = "current") -> ParameterRegistry:
    project = ProjectRegistry.load(project_root)
    rows = [*_fit_rows(project), *_second_continuum_rows(project)]
    frame = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    registry = ParameterRegistry(frame)

    cache_dir = project.root / "data" / "cache" / "parameters"
    cache_dir.mkdir(parents=True, exist_ok=True)
    registry.write(cache_dir / "parameter_registry.csv.gz")

    fit_summary = (
        frame.loc[frame["scope"] == "fit"]
        .groupby(["fit_name", "mixture_id", "model_id", "channel_group"], dropna=False)
        .agg(n_parameters=("name", "count"), n_fixed=("fixed", "sum"), source_file=("source_file", "first"))
        .reset_index()
    )
    fit_summary.to_csv(cache_dir / "fit_summary.csv", index=False)
    registry.comparison_summary().to_csv(cache_dir / "parameter_comparison_summary.csv", index=False)
    requirements = _requirements_report(project, registry)
    requirements.to_csv(cache_dir / "model_parameter_requirements.csv", index=False)
    return registry
