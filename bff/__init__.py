"""Bayesic Force Fields public package namespace."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

__version__ = "0.4.1"

_INTERNAL_MODULES = {
    "bayes",
    "domain",
    "io",
    "plotting",
    "qoi",
    "tools",
}

__all__ = [
    "Project",
    "build",
    "label_snapshots",
    "sample_parameters",
    "build_qoi_datasets",
    "fit_lgp",
    "learn",
    "validate",
    "QoI",
    "QoIDataset",
    "PosteriorResults",
]


def _run_workflow(module_name: str, fn_config: str | Path):
    workflow = import_module(f"{module_name}.main")
    return workflow.main(Path(fn_config))


def build(fn_config: str | Path):
    return _run_workflow("bff.workflows.build", fn_config)


def label_snapshots(fn_config: str | Path):
    return _run_workflow("bff.workflows.label_snapshots", fn_config)


def sample_parameters(fn_config: str | Path):
    return _run_workflow("bff.workflows.sample_parameters", fn_config)


def build_qoi_datasets(fn_config: str | Path):
    return _run_workflow("bff.workflows.build_qoi_datasets", fn_config)


def fit_lgp(fn_config: str | Path):
    return _run_workflow("bff.workflows.fit_lgp", fn_config)


def learn(fn_config: str | Path):
    return _run_workflow("bff.workflows.learn", fn_config)


def validate(fn_config: str | Path):
    return _run_workflow("bff.workflows.validate", fn_config)


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def _resolve(self, fn_config: str | Path) -> Path:
        fn_config = Path(fn_config)
        if fn_config.is_absolute():
            return fn_config
        return (self.root / fn_config).resolve()

    def build(self, fn_config: str | Path):
        return build(self._resolve(fn_config))

    def label_snapshots(self, fn_config: str | Path):
        return label_snapshots(self._resolve(fn_config))

    def sample_parameters(self, fn_config: str | Path):
        return sample_parameters(self._resolve(fn_config))

    def build_qoi_datasets(self, fn_config: str | Path):
        return build_qoi_datasets(self._resolve(fn_config))

    def fit_lgp(self, fn_config: str | Path):
        return fit_lgp(self._resolve(fn_config))

    def learn(self, fn_config: str | Path):
        return learn(self._resolve(fn_config))

    def validate(self, fn_config: str | Path):
        return validate(self._resolve(fn_config))


def __getattr__(name: str) -> Any:
    if name in _INTERNAL_MODULES:
        return import_module(f".{name}", __name__)
    if name == "QoI":
        from .qoi.data import QoI
        return QoI
    if name == "QoIDataset":
        from .qoi.data import QoIDataset
        return QoIDataset
    if name == "PosteriorResults":
        from .bayes.results import PosteriorResults
        return PosteriorResults
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
