from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(slots=True)
class QoI:
    """One named quantity of interest produced for a single trajectory."""

    name: str
    values: np.ndarray
    labels: tuple[str, ...] | None = None
    values_per_label: int = 1
    settings_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).reshape(-1)
        labels = None if self.labels is None else tuple(self.labels)
        values_per_label = int(self.values_per_label)

        if values_per_label <= 0:
            raise ValueError("'values_per_label' must be a positive integer.")
        if labels is not None and len(values) != len(labels) * values_per_label:
            raise ValueError(
                "Number of values does not match labels * values_per_label in QoI."
            )

        self.values = values
        self.labels = labels
        self.values_per_label = values_per_label
        self.settings_kwargs = dict(self.settings_kwargs)
        self.metadata = dict(self.metadata)

    @property
    def n_values(self) -> int:
        """Total number of numeric values stored in the QoI."""
        return int(self.values.size)

    def aligned(self, labels: tuple[str, ...] | None = None) -> np.ndarray:
        """Align the QoI to a reference label set."""
        if labels is None:
            return self.values.copy()

        if self.labels is None:
            expected = len(labels) * self.values_per_label
            if self.values.size != expected:
                raise ValueError(
                    "Unlabeled QoI cannot be aligned to a differently sized "
                    "reference label set."
                )
            return self.values.copy()

        aligned = np.zeros(len(labels) * self.values_per_label, dtype=float)
        index = {label: i for i, label in enumerate(labels)}
        for i, label in enumerate(self.labels):
            if label not in index:
                continue
            src = slice(i * self.values_per_label, (i + 1) * self.values_per_label)
            dst_index = index[label]
            dst = slice(
                dst_index * self.values_per_label,
                (dst_index + 1) * self.values_per_label,
            )
            aligned[dst] = self.values[src]
        return aligned

    def to_dict(self) -> dict[str, Any]:
        """Serialize the QoI to a JSON/PT-friendly mapping."""
        return {
            "name": self.name,
            "values": self.values.tolist(),
            "labels": None if self.labels is None else list(self.labels),
            "values_per_label": self.values_per_label,
            "settings_kwargs": dict(self.settings_kwargs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QoI":
        """Rebuild a serialized QoI object."""
        return cls(
            name=str(data["name"]),
            values=np.asarray(data["values"], dtype=float),
            labels=None if data.get("labels") is None else tuple(data["labels"]),
            values_per_label=int(data.get("values_per_label", 1)),
            settings_kwargs=dict(data.get("settings_kwargs", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class QoIDataset:
    """Training-ready dataset for one named quantity of interest."""

    name: str
    inputs: np.ndarray
    outputs: np.ndarray
    outputs_ref: np.ndarray
    nuisance: float | None = None
    settings_kwargs: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.inputs = np.asarray(self.inputs, dtype=float)
        self.outputs = np.asarray(self.outputs, dtype=float)
        self.outputs_ref = np.asarray(self.outputs_ref, dtype=float).reshape(-1)
        self.settings_kwargs = self._coerce_mapping(self.settings_kwargs)
        self.metadata = self._coerce_mapping(self.metadata)

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError(
                f"Number of input samples ({self.inputs.shape[0]}) does not match "
                f"number of output samples ({self.outputs.shape[0]})."
            )

        if self.outputs.shape[1] != self.outputs_ref.shape[0]:
            raise ValueError(
                f"Output dimension ({self.outputs.shape[1]}) does not match "
                f"reference dimension ({self.outputs_ref.shape[0]})."
            )

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        return dict(value or {})

    @property
    def n_samples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def n_observations(self) -> int:
        values_per_label = int(self.metadata.get("values_per_label", 1) or 1)
        if values_per_label <= 0:
            raise ValueError("'values_per_label' must be a positive integer.")
        if self.outputs_ref.size % values_per_label != 0:
            raise ValueError(
                "Reference output size must be divisible by "
                "'metadata[\"values_per_label\"]'."
            )
        return int(self.outputs_ref.size // values_per_label)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "inputs": self.inputs.tolist(),
            "outputs": self.outputs.tolist(),
            "outputs_ref": self.outputs_ref.tolist(),
            "settings_kwargs": dict(self.settings_kwargs),
            "metadata": dict(self.metadata),
        }
        if self.nuisance is not None:
            data["nuisance"] = self.nuisance
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QoIDataset":
        return cls(
            name=str(data["name"]),
            inputs=np.asarray(data["inputs"], dtype=float),
            outputs=np.asarray(data["outputs"], dtype=float),
            outputs_ref=np.asarray(data["outputs_ref"], dtype=float),
            nuisance=data.get("nuisance"),
            settings_kwargs=dict(data.get("settings_kwargs", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def write(self, fn_out: str) -> None:
        torch.save(self.to_dict(), fn_out)

    @classmethod
    def load(cls, fn_in: str) -> "QoIDataset":
        data = torch.load(fn_in, map_location="cpu", weights_only=False)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"QoIDataset(n_samples={self.n_samples}, "
            f"n_observations={self.n_observations}, name={self.name!r})"
        )
