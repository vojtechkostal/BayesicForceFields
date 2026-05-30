from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linprog
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
        return np.asarray(
            [bounds for _, bounds in self._items],
            dtype=float,
        ).reshape(-1, 2)

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

    def without_names(self, names: Sequence[str]) -> "Bounds":
        excluded = set(names)
        return Bounds({
            key: value for key, value in self.by_name.items() if key not in excluded
        })

    def to_dict(self) -> dict[str, list[float]]:
        return {
            name: [float(lower), float(upper)]
            for name, (lower, upper) in self._items
        }


@dataclass(frozen=True)
class ChargeConstraintSpec:
    """One compiled charge equation stored in ``specs.yaml``."""

    selection: str
    target: float
    scope: str
    implicit: str
    coefficients: Mapping[str, float]
    fixed_charge: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChargeConstraintSpec":
        required = {"selection", "target", "scope", "implicit", "coefficients"}
        missing = required - set(data)
        if missing:
            fields = ", ".join(sorted(repr(key) for key in missing))
            raise ValueError(
                f"Charge constraint is missing required field(s): {fields}"
            )
        scope = str(data["scope"])
        if scope not in {"system", "residue"}:
            raise ValueError(
                f"Unsupported charge-constraint scope {scope!r}; "
                "expected 'system' or 'residue'."
            )
        coefficients = data["coefficients"]
        if not isinstance(coefficients, Mapping):
            raise ValueError("Charge-constraint 'coefficients' must be a mapping.")
        return cls(
            selection=str(data["selection"]),
            target=float(data["target"]),
            scope=scope,
            implicit=str(data["implicit"]),
            coefficients={
                str(name): float(value) for name, value in coefficients.items()
            },
            fixed_charge=float(data.get("fixed_charge", 0.0)),
        )

    @property
    def adjusted_target(self) -> float:
        return self.target - self.fixed_charge

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection,
            "target": self.target,
            "scope": self.scope,
            "implicit": self.implicit,
            "coefficients": dict(self.coefficients),
            "fixed_charge": self.fixed_charge,
        }


@dataclass(frozen=True)
class Specs:
    """Force-field parameters and compiled charge reconstruction rules."""

    source: InitVar[dict[str, Any] | PathLike]

    bounds: Bounds = field(init=False)
    charge_constraints: tuple[ChargeConstraintSpec, ...] = field(init=False)
    implicit_params: tuple[str, ...] = field(init=False)
    reconstruction_order: tuple[int, ...] = field(init=False)

    def __post_init__(self, source: dict[str, Any] | PathLike) -> None:
        if isinstance(source, dict):
            data = dict(source)
        elif isinstance(source, (str, Path)):
            data = load_yaml(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        required = {"bounds", "charge_constraints"}
        missing = required - set(data)
        if missing:
            fields = ", ".join(sorted(repr(key) for key in missing))
            raise ValueError(f"Missing required specs field(s): {fields}")

        raw_constraints = data["charge_constraints"]
        if not isinstance(raw_constraints, list):
            raise ValueError("'charge_constraints' must be a list.")
        if not all(isinstance(item, Mapping) for item in raw_constraints):
            raise ValueError("Each charge constraint must be a mapping.")

        bounds = Bounds(data["bounds"])
        constraints = tuple(
            ChargeConstraintSpec.from_dict(item) for item in raw_constraints
        )
        implicit_params = tuple(constraint.implicit for constraint in constraints)
        if len(implicit_params) != len(set(implicit_params)):
            raise ValueError(
                "Each charge constraint must own a distinct implicit parameter."
            )

        bound_names = set(bounds.names)
        for constraint in constraints:
            if constraint.implicit not in bound_names:
                raise ValueError(
                    f"Implicit parameter {constraint.implicit!r} is not defined "
                    "in bounds."
                )
            if not constraint.implicit.startswith("charge "):
                raise ValueError(
                    f"Implicit parameter {constraint.implicit!r} is not a charge "
                    "parameter."
                )
            unknown = set(constraint.coefficients) - bound_names
            if unknown:
                names = ", ".join(sorted(repr(name) for name in unknown))
                raise ValueError(
                    f"Charge constraint {constraint.selection!r} references "
                    f"unknown bounded parameter(s): {names}."
                )
            invalid = [
                name
                for name in constraint.coefficients
                if not name.startswith("charge ")
            ]
            if invalid:
                names = ", ".join(sorted(repr(name) for name in invalid))
                raise ValueError(
                    f"Charge constraint {constraint.selection!r} references "
                    f"non-charge parameter(s): {names}."
                )
            if np.isclose(constraint.coefficients.get(constraint.implicit, 0.0), 0.0):
                raise ValueError(
                    f"Implicit parameter {constraint.implicit!r} is not selected by "
                    f"its owning constraint {constraint.selection!r}."
                )

        owners = {name: i for i, name in enumerate(implicit_params)}
        dependencies = {
            i: {
                owners[name]
                for name, coefficient in constraint.coefficients.items()
                if (
                    name in owners
                    and name != constraint.implicit
                    and not np.isclose(coefficient, 0.0)
                )
            }
            for i, constraint in enumerate(constraints)
        }
        order: list[int] = []
        pending = set(range(len(constraints)))
        while pending:
            ready = sorted(i for i in pending if dependencies[i] <= set(order))
            if not ready:
                raise ValueError(
                    "Charge constraints contain a cyclic implicit-parameter dependency."
                )
            order.extend(ready)
            pending.difference_update(ready)

        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "charge_constraints", constraints)
        object.__setattr__(self, "implicit_params", implicit_params)
        object.__setattr__(self, "reconstruction_order", tuple(order))
        self._check_feasibility()

    @classmethod
    def load(cls, source: PathLike) -> "Specs":
        return cls(source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": self.bounds.to_dict(),
            "charge_constraints": [
                constraint.to_dict() for constraint in self.charge_constraints
            ],
        }

    def write(self, fn_out: PathLike) -> None:
        save_yaml(self.to_dict(), fn_out)

    @property
    def explicit_bounds(self) -> Bounds:
        return self.bounds.without_names(self.implicit_params)

    @property
    def constraint_matrix(self) -> np.ndarray:
        return np.asarray([
            [constraint.coefficients.get(name, 0.0) for name in self.bounds.names]
            for constraint in self.charge_constraints
        ], dtype=float)

    @property
    def constraint_targets(self) -> np.ndarray:
        return np.asarray([
            constraint.adjusted_target for constraint in self.charge_constraints
        ], dtype=float)

    def _check_feasibility(self) -> None:
        if not self.charge_constraints:
            return
        result = linprog(
            np.zeros(self.bounds.n_params),
            A_eq=self.constraint_matrix,
            b_eq=self.constraint_targets,
            bounds=self.bounds.array,
            method="highs",
        )
        if not result.success:
            raise ValueError(
                "Charge constraints are incompatible with each other or with the "
                f"configured bounds: {result.message}"
            )

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

    def with_implicit_charges(
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

        names = self.bounds.names.tolist()
        full = np.zeros((len(array), self.bounds.n_params), dtype=float)
        explicit_indices = [names.index(name) for name in self.explicit_bounds.names]
        full[:, explicit_indices] = array[:, :n_explicit]
        for index in self.reconstruction_order:
            constraint = self.charge_constraints[index]
            implicit_index = names.index(constraint.implicit)
            coefficient = constraint.coefficients[constraint.implicit]
            full[:, implicit_index] = (
                constraint.adjusted_target - full @ self.constraint_matrix[index]
            ) / coefficient

        return np.concatenate([full, array[:, n_explicit:]], axis=1)


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
    def explicit_parameter_names(self) -> list[str]:
        return self.specs.explicit_bounds.names.tolist()

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

        full = self.specs.with_implicit_charges(x)
        lower, upper = self.specs.bounds.array.T
        return ((full >= lower) & (full <= upper)).all(axis=1)

    def __call__(self, values: ArrayLike) -> np.ndarray | Any:
        valid = self.is_valid(values)
        if _is_torch_tensor(values):
            return torch.as_tensor(valid, device=values.device)
        return valid


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
        self.sampler = (
            None if self.bounds.shape[0] == 0 else LatinHypercube(self.bounds.shape[0])
        )

    def __call__(self, n: int) -> np.ndarray:
        if n < 0:
            raise ValueError("Number of samples must be non-negative.")
        if n == 0:
            return np.empty((0, self.bounds.shape[0]), dtype=float)
        if self.bounds.shape[0] == 0:
            samples = np.empty((n, 0), dtype=float)
            if self.constraint is not None and not np.asarray(
                self.constraint(samples),
                dtype=bool,
            ).all():
                raise RuntimeError("Fully constrained parameter values are invalid.")
            self.n_generated += n
            return samples

        lower, upper = self.bounds.T
        if self.constraint is None:
            assert self.sampler is not None
            unit_samples = self.sampler.random(n)
            self.n_generated += n
            return unit_samples * (upper - lower) + lower

        collected: list[np.ndarray] = []
        n_valid = 0
        attempts = 0
        while n_valid < n:
            assert self.sampler is not None
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
