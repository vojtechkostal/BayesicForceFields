"""QoI blocks from one trajectory and training-ready QoI datasets."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..io.utils import load_pt, save_pt


@dataclass(slots=True)
class QoI:
    """One named quantity of interest produced for a single trajectory."""

    name: str
    values: np.ndarray
    labels: tuple[str, ...] | None = None
    values_per_label: int = 1
    settings: dict[str, Any] = field(default_factory=dict)
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
        if not np.all(np.isfinite(values)):
            raise ValueError("QoI values must all be finite.")

        self.values = values
        self.labels = labels
        self.values_per_label = values_per_label
        self.settings = dict(self.settings)
        self.metadata = dict(self.metadata)

    @property
    def n_values(self) -> int:
        """Total number of numeric values stored in the QoI."""
        return int(self.values.size)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the QoI to a JSON/PT-friendly mapping."""
        return {
            "name": self.name,
            "values": self.values.tolist(),
            "labels": None if self.labels is None else list(self.labels),
            "values_per_label": self.values_per_label,
            "settings": dict(self.settings),
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
            settings=dict(data.get("settings", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class QoIDataset:
    """Training-ready dataset for one named quantity of interest.

    ``X`` holds one parameter vector per sample, ``y`` the sample's QoI
    values, and ``y_ref`` the reference values; ``sample_ids`` and
    ``parameter_names`` name the rows and columns of ``X``.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    y_ref: np.ndarray
    labels: tuple[str, ...] | None = None
    values_per_label: int = 1
    nuisance: float | None = None
    settings: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    sample_ids: tuple[str, ...] | None = None
    parameter_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        if self.sample_ids is not None:
            self.sample_ids = tuple(str(value) for value in self.sample_ids)
        if self.parameter_names is not None:
            self.parameter_names = tuple(str(value) for value in self.parameter_names)
        self.y = np.asarray(self.y, dtype=float)
        self.y_ref = np.asarray(self.y_ref, dtype=float).reshape(-1)
        self.settings = dict(self.settings or {})
        self.metadata = dict(self.metadata or {})
        self.labels = None if self.labels is None else tuple(self.labels)
        self.values_per_label = int(self.values_per_label)

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X has {self.X.shape[0]} rows but y has {self.y.shape[0]}."
            )
        for field_name, values in (
            ("X", self.X),
            ("y", self.y),
            ("y_ref", self.y_ref),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"QoIDataset {field_name} must all be finite.")

        if self.sample_ids is not None and len(self.sample_ids) != self.n_samples:
            raise ValueError(
                f"{len(self.sample_ids)} sample_ids for {self.n_samples} samples."
            )
        if self.parameter_names is not None and len(self.parameter_names) != (
            self.X.shape[1]
        ):
            raise ValueError(
                f"{len(self.parameter_names)} parameter_names for "
                f"{self.X.shape[1]} columns of X."
            )
        if self.y.shape[1] != self.y_ref.shape[0]:
            raise ValueError(
                f"Output dimension ({self.y.shape[1]}) does not match "
                f"reference dimension ({self.y_ref.shape[0]})."
            )
        if self.values_per_label <= 0:
            raise ValueError("'values_per_label' must be a positive integer.")
        if (
            self.labels is not None
            and self.y_ref.size != len(self.labels) * self.values_per_label
        ):
            raise ValueError(
                "Reference output size does not match labels * values_per_label."
            )

    @property
    def n_samples(self) -> int:
        """Number of samples (rows of ``X``)."""
        return int(self.X.shape[0])

    @property
    def n_curves(self) -> int:
        """Number of curves (labels) that ``y_ref`` concatenates."""
        if self.y_ref.size % self.values_per_label != 0:
            raise ValueError(
                "Reference output size must be divisible by "
                "'values_per_label'."
            )
        if self.labels is not None:
            return len(self.labels)
        return int(self.y_ref.size // self.values_per_label)

    @property
    def curve_length(self) -> int:
        """Number of values per curve."""
        return int(self.y_ref.size // self.n_curves)

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping of the dataset, readable by :meth:`from_dict`."""
        data = {
            "name": self.name,
            "X": self.X.tolist(),
            "y": self.y.tolist(),
            "y_ref": self.y_ref.tolist(),
            "labels": None if self.labels is None else list(self.labels),
            "values_per_label": self.values_per_label,
            "settings": dict(self.settings),
            "metadata": dict(self.metadata),
            "sample_ids": None if self.sample_ids is None else list(self.sample_ids),
            "parameter_names": (
                None if self.parameter_names is None else list(self.parameter_names)
            ),
        }
        if self.nuisance is not None:
            data["nuisance"] = self.nuisance
        return data

    def fingerprint(self) -> str:
        """Return a deterministic fingerprint of all model-training inputs."""
        digest = hashlib.sha256()
        metadata = {
            "name": self.name,
            "labels": self.labels,
            "values_per_label": self.values_per_label,
            "nuisance": self.nuisance,
            "settings": self.settings,
            "metadata": self.metadata,
            "sample_ids": self.sample_ids,
            "parameter_names": self.parameter_names,
        }
        digest.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for values in (self.X, self.y, self.y_ref):
            array = np.ascontiguousarray(values, dtype=np.float64)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QoIDataset":
        """Inverse of :meth:`to_dict`."""
        return cls(
            name=str(data["name"]),
            X=np.asarray(data["X"], dtype=float),
            y=np.asarray(data["y"], dtype=float),
            y_ref=np.asarray(data["y_ref"], dtype=float),
            labels=None if data.get("labels") is None else tuple(data["labels"]),
            values_per_label=int(data.get("values_per_label", 1)),
            nuisance=data.get("nuisance"),
            settings=dict(data.get("settings", {})),
            metadata=dict(data.get("metadata", {})),
            sample_ids=data.get("sample_ids"),
            parameter_names=data.get("parameter_names"),
        )

    def write(self, fn_out: str) -> None:
        """Write the dataset to a ``.pt`` file."""
        save_pt(self.to_dict(), fn_out)

    @classmethod
    def load(cls, fn_in: str) -> "QoIDataset":
        """Read a ``.pt`` file written by :meth:`write`."""
        data = load_pt(fn_in)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"QoIDataset({self.name!r}, n_samples={self.n_samples}, "
            f"n_curves={self.n_curves}, curve_length={self.curve_length})"
        )
