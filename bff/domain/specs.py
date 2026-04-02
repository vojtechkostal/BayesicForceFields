from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats.qmc import LatinHypercube

from ..io.utils import load_yaml, save_yaml

try:
    import torch
except ModuleNotFoundError:
    torch = None


PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, Any]


def _is_torch_tensor(values: object) -> bool:
    return torch is not None and isinstance(values, torch.Tensor)


@dataclass(frozen=True)
class Bounds:
    """Named parameter bounds with a stable sorted order."""

    by_name: Mapping[str, Tuple[float, float]]
    _items: tuple[tuple[str, tuple[float, float]], ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        items = tuple(sorted(self.by_name.items()))
        for name, (lower, upper) in items:
            if lower > upper:
                raise ValueError(
                    f"Lower bound {lower} is greater than upper bound {upper} "
                    f"for parameter {name!r}."
                )
        object.__setattr__(self, "_items", items)

    @property
    def names(self) -> np.ndarray:
        return np.array([name for name, _ in self._items], dtype=str)

    @property
    def array(self) -> np.ndarray:
        return np.asarray([bounds for _, bounds in self._items], dtype=float)

    @property
    def lower(self) -> np.ndarray:
        return self.array[:, 0]

    @property
    def upper(self) -> np.ndarray:
        return self.array[:, 1]

    @property
    def n_params(self) -> int:
        return len(self._items)

    def get(self, name: str) -> tuple[float, float]:
        return self.by_name[name]

    def index(self, name: str) -> int:
        names = self.names
        mask = names == name
        if not np.any(mask):
            raise ValueError(f"Parameter {name!r} not found in bounds.")
        return int(np.argwhere(mask).ravel()[0])

    def without(self, name: str) -> "Bounds":
        items = {
            key: value for key, value in self.by_name.items() if key != name
        }
        return Bounds(items)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            name: [float(lower), float(upper)]
            for name, (lower, upper) in self._items
        }


@dataclass(frozen=True)
class Specs:
    """Force-field specification data."""

    source: InitVar[dict[str, Any] | PathLike]

    mol_resname: str = field(init=False)
    bounds: Bounds = field(init=False)
    total_charge: float = field(init=False)
    constraint_charge: float = field(init=False)
    implicit_atoms: tuple[str, ...] = field(init=False)

    def __post_init__(self, source: dict[str, Any] | PathLike) -> None:
        if isinstance(source, dict):
            data = dict(source)
        elif isinstance(source, (str, Path)):
            data = load_yaml(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        required = {"bounds", "implicit_atoms", "charge_target"}
        missing = required - set(data)
        if missing:
            missing_list = ", ".join(sorted(repr(key) for key in missing))
            raise ValueError(f"Missing required specs field(s): {missing_list}")

        implicit_atoms = data["implicit_atoms"]
        if isinstance(implicit_atoms, str):
            implicit_atoms = implicit_atoms.split()

        object.__setattr__(self, "mol_resname", str(data.get("mol_resname", "")))
        object.__setattr__(self, "bounds", Bounds(data["bounds"]))
        object.__setattr__(self, "total_charge", float(data.get("total_charge", 0.0)))
        object.__setattr__(self, "constraint_charge", float(data["charge_target"]))
        object.__setattr__(self, "implicit_atoms", tuple(implicit_atoms))

    @classmethod
    def load(cls, source: PathLike) -> "Specs":
        return cls(source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mol_resname": self.mol_resname,
            "bounds": self.bounds.to_dict(),
            "total_charge": self.total_charge,
            "charge_target": self.constraint_charge,
            "implicit_atoms": list(self.implicit_atoms),
        }

    def write(self, fn_out: PathLike) -> None:
        save_yaml(self.to_dict(), fn_out)

    @property
    def implicit_param(self) -> str:
        return f"charge {' '.join(self.implicit_atoms)}"

    @property
    def implicit_param_index(self) -> int:
        return self.bounds.index(self.implicit_param)

    @property
    def explicit_bounds(self) -> Bounds:
        return self.bounds.without(self.implicit_param)

    def parameter_names(self, *, explicit_only: bool = False) -> tuple[str, ...]:
        bounds = self.explicit_bounds if explicit_only else self.bounds
        return tuple(str(name) for name in bounds.names)

    def parameter_dict(
        self,
        values: Sequence[float] | np.ndarray | Any,
        *,
        explicit_only: bool = False,
    ) -> dict[str, float]:
        array = (
            values.detach().cpu().numpy()
            if _is_torch_tensor(values)
            else np.asarray(values, dtype=float)
        ).reshape(-1)
        names = self.parameter_names(explicit_only=explicit_only)
        if array.size != len(names):
            raise ValueError(
                f"Expected {len(names)} parameter values, got {array.size}."
            )
        return {name: float(value) for name, value in zip(names, array)}

    @property
    def explicit_charge_coefficients(self) -> np.ndarray:
        coeffs: list[int] = []
        for param in self.explicit_bounds.names:
            atoms = param.split()[1:]
            coeffs.append(len(atoms) if param.startswith("charge") else 0)
        return np.asarray(coeffs, dtype=int)

    @property
    def constraint_matrix(self) -> np.ndarray:
        return self.explicit_charge_coefficients

    def implicit_charge(
        self,
        values: Sequence[float] | np.ndarray | Any,
    ) -> np.ndarray:
        array = (
            values.detach().cpu().numpy()
            if _is_torch_tensor(values)
            else np.asarray(values, dtype=float)
        )
        array = np.atleast_2d(array)
        n_explicit = self.explicit_bounds.n_params
        if array.shape[1] < n_explicit:
            raise ValueError(
                f"Expected at least {n_explicit} explicit parameter columns, "
                f"got {array.shape[1]}."
            )
        explicit = array[:, :n_explicit]
        explicit_charge = explicit @ self.constraint_matrix
        return (self.constraint_charge - explicit_charge) / len(self.implicit_atoms)

    def with_implicit_charge(
        self,
        values: Sequence[float] | np.ndarray | Any,
    ) -> np.ndarray:
        array = (
            values.detach().cpu().numpy()
            if _is_torch_tensor(values)
            else np.asarray(values, dtype=float)
        )
        array = np.atleast_2d(array)
        n_explicit = self.explicit_bounds.n_params
        if array.shape[1] < n_explicit:
            raise ValueError(
                f"Expected at least {n_explicit} explicit parameter columns, "
                f"got {array.shape[1]}."
            )

        implicit = self.implicit_charge(array)[:, None]
        insert_at = self.implicit_param_index
        return np.concatenate(
            [
                array[:, :insert_at],
                implicit,
                array[:, insert_at:],
            ],
            axis=1,
        )


@dataclass(frozen=True)
class ChargeConstraint:
    """Charge-aware validity check derived from ``Specs``."""

    specs: Specs

    def __init__(self, specs: Specs | dict[str, Any] | PathLike) -> None:
        object.__setattr__(
            self,
            "specs",
            specs if isinstance(specs, Specs) else Specs(specs),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.specs, name)

    @property
    def explicit_bounds(self) -> np.ndarray:
        return self.specs.explicit_bounds.array

    @property
    def implicit_bounds(self) -> tuple[float, float]:
        return self.specs.bounds.get(self.specs.implicit_param)

    @property
    def n_params(self) -> int:
        return self.specs.explicit_bounds.n_params

    @staticmethod
    def _to_2d_array(values: ArrayLike) -> np.ndarray:
        if _is_torch_tensor(values):
            arr = values.detach().cpu().numpy()
        else:
            arr = np.asarray(values, dtype=float)
        return np.atleast_2d(arr)

    def is_valid(self, values: ArrayLike) -> np.ndarray:
        x = self._to_2d_array(values)
        if x.shape[1] != self.n_params:
            raise ValueError(
                f"Input array must have shape (n_samples, {self.n_params}), "
                f"but got {x.shape}."
            )

        explicit_lower, explicit_upper = self.explicit_bounds.T
        implicit_lower, implicit_upper = self.implicit_bounds

        in_explicit_bounds = ((x >= explicit_lower) & (x <= explicit_upper)).all(axis=1)
        explicit_total_charge = np.sum(
            x * self.specs.explicit_charge_coefficients,
            axis=1,
        )
        implicit_charge = self.specs.constraint_charge - explicit_total_charge
        in_implicit_bounds = (
            (implicit_charge >= implicit_lower) & (implicit_charge <= implicit_upper)
        )
        return in_explicit_bounds & in_implicit_bounds

    def __call__(self, values: ArrayLike) -> np.ndarray:
        return self.is_valid(values)


class RandomParamsGenerator:
    """Latin-hypercube parameter generator with optional validity filter."""

    def __init__(
        self,
        bounds: np.ndarray,
        constraint: Optional[Callable[[ArrayLike], np.ndarray]] = None,
    ) -> None:
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds must have shape (n_params, 2).")

        self.bounds = np.asarray(bounds, dtype=float)
        self.constraint = constraint
        self.n_generated = 0
        self.sampler = LatinHypercube(self.bounds.shape[0])

    def __call__(self, n: int) -> np.ndarray:
        if n < 0:
            raise ValueError("Number of samples must be non-negative.")
        if n == 0:
            return np.empty((0, self.bounds.shape[0]), dtype=float)

        lower, upper = self.bounds.T
        if self.constraint is None:
            unit_samples = self.sampler.random(n)
            self.n_generated += n
            return unit_samples * (upper - lower) + lower

        collected: list[np.ndarray] = []
        n_valid = 0
        attempts = 0
        while n_valid < n:
            batch_size = max(2 * (n - n_valid), 1)
            unit_samples = self.sampler.random(batch_size)
            self.n_generated += batch_size
            attempts += 1

            samples = unit_samples * (upper - lower) + lower
            mask = np.asarray(self.constraint(samples), dtype=bool).reshape(-1)
            valid = samples[mask]
            if valid.size > 0:
                collected.append(valid)
                n_valid += len(valid)
                attempts = 0

            if attempts >= 1000:
                raise RuntimeError(
                    "Failed to generate valid parameter samples within 1000 "
                    "consecutive Latin-hypercube batches."
                )

        return np.vstack(collected)[:n]
