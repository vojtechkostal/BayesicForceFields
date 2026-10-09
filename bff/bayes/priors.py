"""Priors of the sampled parameters and nuisance parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Uniform


@dataclass(frozen=True, slots=True)
class Prior:
    """One prior: ``normal`` (``a`` = mean, ``b`` = standard deviation) or
    ``uniform`` (``a`` = lower, ``b`` = upper bound)."""

    kind: str
    a: float
    b: float
    name: Optional[str] = None

    def __post_init__(self) -> None:
        kind = self.kind.lower()
        if kind not in {"normal", "uniform"}:
            raise ValueError(
                f'Unknown prior type "{self.kind}". Options are "normal" or "uniform".'
            )
        if kind == "normal" and self.b <= 0:
            raise ValueError("Normal prior scale must be positive.")
        if kind == "uniform" and self.a >= self.b:
            raise ValueError("Uniform prior requires lower < upper.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "a", float(self.a))
        object.__setattr__(self, "b", float(self.b))

    @property
    def distribution(self) -> Distribution:
        """The torch distribution."""
        if self.kind == "normal":
            return Normal(self.a, self.b, validate_args=False)
        return Uniform(self.a, self.b, validate_args=False)

    @property
    def mean(self) -> float:
        """Mean of the prior."""
        return self.a if self.kind == "normal" else 0.5 * (self.a + self.b)

    @property
    def scale(self) -> float:
        """Standard deviation."""
        return self.b if self.kind == "normal" else (self.b - self.a) / np.sqrt(12)

    def to_dict(self) -> dict[str, float | str]:
        """Plain mapping, readable by ``Prior(**data)``."""
        data = {"kind": self.kind, "a": self.a, "b": self.b}
        if self.name is not None:
            data["name"] = self.name
        return data


@dataclass(slots=True)
class Priors:
    """The priors of all dimensions of a chain, in column order."""

    items: list[Prior] = field(default_factory=list)
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __repr__(self) -> str:
        return f"Priors({len(self.items)}: {', '.join(self.names)})"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> Prior:
        return self.items[index]

    @property
    def names(self) -> list[str]:
        """Prior names; unnamed priors are ``theta_<column>``."""
        return [prior.name or f"theta_{i}" for i, prior in enumerate(self.items)]

    @property
    def distributions(self) -> list[Distribution]:
        """The torch distributions, in column order."""
        return [prior.distribution for prior in self.items]

    @property
    def means(self) -> np.ndarray:
        """Prior means, shape ``(n_dim,)``."""
        return np.array([prior.mean for prior in self.items], dtype=float)

    @property
    def scales(self) -> np.ndarray:
        """Prior standard deviations, shape ``(n_dim,)``."""
        return np.array([prior.scale for prior in self.items], dtype=float)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log prior of each row of ``theta``, on the device of ``theta``."""
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        is_normal, a, b = self._tensors(theta.device, theta.dtype)
        normal = -0.5 * ((theta - a) / b).pow(2) - b.log() - 0.5 * math.log(2 * math.pi)
        inside = (theta >= a) & (theta <= b)
        uniform = torch.where(inside, -(b - a).log(), -torch.inf)
        return torch.where(is_normal, normal, uniform).sum(dim=1)

    def _tensors(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = (
                torch.tensor([p.kind == "normal" for p in self.items], device=device),
                torch.tensor([p.a for p in self.items], device=device, dtype=dtype),
                torch.tensor([p.b for p in self.items], device=device, dtype=dtype),
            )
        return self._cache[key]

    def to_dicts(self) -> list[dict[str, float | str]]:
        """One :meth:`Prior.to_dict` per column."""
        return [prior.to_dict() for prior in self.items]

    @classmethod
    def from_dicts(cls, records: Sequence[dict]) -> Priors:
        """Inverse of :meth:`to_dicts`."""
        return cls([Prior(**record) for record in records])

    @classmethod
    def from_bounds(
        cls,
        bounds: np.ndarray,
        dist_type: str = "normal",
        names: Optional[Sequence[str]] = None,
        nuisance_names: Sequence[str] = (),
    ) -> Priors:
        """Priors over parameter bounds, then a normal prior of -2 +- 2 (a
        log standard deviation) for each nuisance parameter.

        ``normal`` priors are centred on each interval with a standard
        deviation of a fifth of its width; ``uniform`` priors cover it.
        """
        bounds = np.asarray(bounds, dtype=float).reshape(-1, 2)
        dist_type = dist_type.lower()
        if names is None:
            names = [None] * len(bounds)
        elif len(names) != len(bounds):
            raise ValueError("names must match the number of parameter priors.")
        if dist_type == "normal":
            items = [
                Prior("normal", 0.5 * (lower + upper), 0.2 * (upper - lower), name)
                for (lower, upper), name in zip(bounds, names)
            ]
        elif dist_type == "uniform":
            items = [
                Prior("uniform", lower, upper, name)
                for (lower, upper), name in zip(bounds, names)
            ]
        else:
            raise ValueError(
                f'Unknown prior type "{dist_type}". Options are "normal" or "uniform".'
            )
        items += [Prior("normal", -2.0, 2.0, name) for name in nuisance_names]
        return cls(items)
