"""Prior means of the local Gaussian processes.

A Gaussian process predicts ``mean(X)`` plus a correction learned from the
training samples; far from the samples only ``mean(X)`` remains, so the mean
is what the surrogate falls back to. A mean is specified as

- ``"data"`` (default): the average training output, per output value;
- a number: that constant for every output value;
- ``"sigmoid"``: for RDF datasets, a smooth step from 0 to 1 in each curve,
  centred where the reference RDF first reaches 0.5;
- ``"module:function"`` or ``"path/to/file.py:function"``: a custom mean;
- any Python callable (Python API only; saved models pickle it).

A custom mean is a function ``mean(X)`` of a tensor of parameter vectors,
shape ``(n_samples, n_parameters)``, that returns the outputs, shape
``(n_samples, n_outputs)`` or ``(n_outputs,)``, preferably as a torch
tensor so that it runs on the model's device.
"""

from __future__ import annotations

from typing import Any, Callable, Union

import numpy as np
import torch

from ..qoi.dataset import QoIDataset
from ..qoi.routines import load_custom_routine

MeanSpec = Union[str, float, Callable[[torch.Tensor], Any], None]
Mean = Union[torch.Tensor, Callable[[torch.Tensor], Any]]

# Steepness of the RDF sigmoid, per unit of distance (1/angstrom for RDFs).
SIGMOID_STEEPNESS = 5.0


class ImportedMean:
    """A custom mean given as ``module:function`` or ``path.py:function``.

    Saved models store only the reference, so they load wherever the
    function can be imported.
    """

    def __init__(self, specification: str) -> None:
        self.specification, self.function = load_custom_routine(specification)

    def __call__(self, X: torch.Tensor) -> Any:
        return self.function(X)

    def __getstate__(self) -> dict[str, str]:
        return {"specification": self.specification}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.__init__(state["specification"])

    def __repr__(self) -> str:
        return f"ImportedMean({self.specification!r})"


def rdf_sigmoid_mean(dataset: QoIDataset) -> np.ndarray:
    """A 0-to-1 step per RDF curve where its reference first reaches 0.5."""
    bins = dataset.settings.get("bins")
    distance_range = dataset.settings.get("range")
    if bins is None or distance_range is None:
        raise ValueError(
            f"The sigmoid mean of {dataset.name!r} needs RDF settings 'bins' and "
            "'range' in the dataset; build it with one RDF routine definition."
        )
    if bins != dataset.curve_length:
        raise ValueError(
            f"{dataset.name!r} declares bins={bins!r}, but each curve has "
            f"{dataset.curve_length} values."
        )
    r0, r1 = distance_range
    r = np.linspace(r0, r1, bins, endpoint=False) + (r1 - r0) / (2 * bins)
    curves = []
    for reference in dataset.y_ref.reshape(-1, bins):
        rising = np.flatnonzero(reference >= 0.5)
        center = r[rising[0]] if rising.size else 0.5 * (r0 + r1)
        curves.append(1.0 / (1.0 + np.exp(-SIGMOID_STEEPNESS * (r - center))))
    return np.concatenate(curves)


def build_mean(spec: MeanSpec, X_train: torch.Tensor, y_train: torch.Tensor) -> Mean:
    """Turn a mean specification into a tensor or callable for the GP.

    ``"sigmoid"`` needs the dataset and is resolved by :func:`resolve_spec`
    first; here it arrives as an array.
    """
    n_outputs = y_train.shape[1]
    if spec is None or (isinstance(spec, str) and spec == "data"):
        return y_train.mean(dim=0)
    if isinstance(spec, str):
        if ":" not in spec:
            raise ValueError(
                f"Unknown mean {spec!r}; use 'data', 'sigmoid', a number, or "
                "'module:function'."
            )
        spec = ImportedMean(spec)
    if callable(spec):
        values = evaluate_mean(spec, X_train)
        if values.shape != y_train.shape:
            raise ValueError(
                f"The mean returns shape {tuple(values.shape)} for "
                f"{len(X_train)} samples; expected {tuple(y_train.shape)}."
            )
        return spec
    values = torch.as_tensor(np.asarray(spec, dtype=float), dtype=y_train.dtype)
    values = values.to(y_train.device).expand(n_outputs).clone()
    return values


def resolve_spec(spec: MeanSpec, dataset: QoIDataset) -> MeanSpec:
    """Replace the dataset-dependent ``"sigmoid"`` by its values."""
    return rdf_sigmoid_mean(dataset) if spec == "sigmoid" else spec


def describe_spec(spec: MeanSpec) -> str | None:
    """The specification as text, or ``None`` for a Python callable."""
    if spec is None:
        return "data"
    if isinstance(spec, (str, int, float)) and not isinstance(spec, bool):
        return str(spec)
    return None


def evaluate_mean(mean: Mean, X: torch.Tensor) -> torch.Tensor:
    """Mean outputs at ``X``, shape ``(n_samples, n_outputs)``."""
    values = mean(X) if callable(mean) else mean
    values = torch.as_tensor(values, dtype=X.dtype, device=X.device)
    return values.expand(len(X), -1) if values.dim() == 1 else values
