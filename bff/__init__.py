"""Bayesic Force Fields public package namespace."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

__version__ = "0.2.0"

_INTERNAL_MODULES = {
    "bayes",
    "data",
    "domain",
    "io",
    "plotting",
    "qoi",
    "tools",
}

__all__ = [
    "Project",
    "build",
    "prepare_assets",
    "evaluate_snapshots",
    "sample",
    "analyze",
    "fit",
    "learn",
    "validate",
    "QoI",
    "QoIDataset",
    "PosteriorResults",
]


def _run_workflow(module_name: str, fn_config: str | Path):
    workflow = import_module(module_name)
    return workflow.main(Path(fn_config))


def build(fn_config: str | Path):
    return _run_workflow("bff.workflows.build", fn_config)


def prepare_assets(fn_config: str | Path):
    return _run_workflow("bff.workflows.prepare_assets", fn_config)


def evaluate_snapshots(fn_config: str | Path):
    return _run_workflow("bff.workflows.evaluate_snapshots", fn_config)


def sample(fn_config: str | Path):
    return _run_workflow("bff.workflows.sample", fn_config)


def analyze(fn_config: str | Path):
    return _run_workflow("bff.workflows.analyze", fn_config)


def fit(fn_config: str | Path):
    return _run_workflow("bff.workflows.fit", fn_config)


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

    def prepare_assets(self, fn_config: str | Path):
        return prepare_assets(self._resolve(fn_config))

    def evaluate_snapshots(self, fn_config: str | Path):
        return evaluate_snapshots(self._resolve(fn_config))

    def sample(self, fn_config: str | Path):
        return sample(self._resolve(fn_config))

    def analyze(self, fn_config: str | Path):
        return analyze(self._resolve(fn_config))

    def fit(self, fn_config: str | Path):
        return fit(self._resolve(fn_config))

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
