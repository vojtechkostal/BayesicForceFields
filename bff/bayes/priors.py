from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Uniform

ArrayLike = Union[np.ndarray, torch.Tensor]
PathLike = Union[str, Path]


@dataclass(frozen=True, slots=True)
class Prior:
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
        if self.kind == "normal":
            return Normal(self.a, self.b, validate_args=False)
        return Uniform(self.a, self.b, validate_args=False)

    @property
    def mean(self) -> float:
        return self.a if self.kind == "normal" else 0.5 * (self.a + self.b)

    @property
    def scale(self) -> float:
        return self.b if self.kind == "normal" else (self.b - self.a) / np.sqrt(12)

    def to_dict(self) -> dict[str, float | str]:
        data = {"kind": self.kind, "a": self.a, "b": self.b}
        if self.name is not None:
            data["name"] = self.name
        return data


@dataclass(slots=True)
class Priors:
    items: list[Prior] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx: int) -> Prior:
        return self.items[idx]

    @property
    def names(self) -> list[str]:
        return [prior.name or f"theta_{i}" for i, prior in enumerate(self.items)]

    @property
    def distributions(self) -> list[Distribution]:
        return [prior.distribution for prior in self.items]

    @property
    def means(self) -> np.ndarray:
        return np.array([prior.mean for prior in self.items], dtype=float)

    @property
    def scales(self) -> np.ndarray:
        return np.array([prior.scale for prior in self.items], dtype=float)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        log_probs = [
            prior.distribution.log_prob(theta[:, i]).to(theta.device)
            for i, prior in enumerate(self.items)
        ]
        return torch.stack(log_probs, dim=1).sum(dim=1)

    def write(self, fn_out: PathLike) -> None:
        fn_out = Path(fn_out)
        data = {"priors": [prior.to_dict() for prior in self.items]}
        torch.save(data, fn_out)

    @classmethod
    def load(cls, fn_in: PathLike) -> "Priors":
        raw = torch.load(Path(fn_in), weights_only=False)
        entries = raw["priors"] if isinstance(raw, dict) and "priors" in raw else raw
        return cls([cls._coerce_prior(entry) for entry in entries])

    @classmethod
    def from_any(cls, priors: Union["Priors", PathLike, Iterable]) -> "Priors":
        if isinstance(priors, cls):
            return priors
        if isinstance(priors, (str, Path)):
            return cls.load(priors)
        return cls([cls._coerce_prior(entry) for entry in priors])

    @staticmethod
    def _coerce_prior(entry) -> Prior:
        if isinstance(entry, Prior):
            return entry

        if isinstance(entry, Distribution):
            name = type(entry).__name__.lower()
            if name == "normal":
                return Prior("normal", float(entry.mean), float(entry.scale))
            if name == "uniform":
                return Prior("uniform", float(entry.low), float(entry.high))
            raise ValueError(f"Unsupported distribution type: {name}")

        if isinstance(entry, dict):
            return Prior(
                kind=entry["kind"],
                a=entry["a"],
                b=entry["b"],
                name=entry.get("name"),
            )

        raise TypeError(f"Unsupported prior entry type: {type(entry)}")

    @classmethod
    def from_bounds(
        cls,
        bounds: ArrayLike,
        dist_type: str = "normal",
        n_nuisance: int = 0,
        names: Optional[Sequence[str]] = None,
        nuisance_names: Optional[Sequence[str]] = None,
    ) -> "Priors":
        bounds = np.asarray(bounds, dtype=float)
        dist_type = dist_type.lower()

        if names is None:
            names = [None] * len(bounds)
        elif len(names) != len(bounds):
            raise ValueError("names must match the number of parameter priors.")

        if dist_type == "normal":
            centers = bounds.mean(axis=1)
            widths = 0.2 * (bounds[:, 1] - bounds[:, 0])
            items = [
                Prior("normal", center, width, name=name)
                for center, width, name in zip(centers, widths, names)
            ]
        elif dist_type == "uniform":
            items = [
                Prior("uniform", lower, upper, name=name)
                for (lower, upper), name in zip(bounds, names)
            ]
        else:
            raise ValueError(
                f'Unknown prior type "{dist_type}". Options are "normal" or "uniform".'
            )

        nuisance_names = nuisance_names or [f"nuisance_{i}" for i in range(n_nuisance)]
        items.extend(Prior("normal", -2.0, 2.0, name=name) for name in nuisance_names)
        return cls(items)


def log_prior(
    theta: torch.Tensor,
    priors: Union[Priors, Sequence[Distribution]],
) -> torch.Tensor:
    return Priors.from_any(priors).log_prob(theta)
